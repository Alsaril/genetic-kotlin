package com.alsaril

import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.special.Erf
import org.apache.commons.math3.util.FastMath

interface ArgumentProvider {
    fun get(name: String): Double?
}

object EmptyVariableProvider : ArgumentProvider {
    override fun get(name: String) = null
}

class FallbackProvider(private vararg val provider: ArgumentProvider) : ArgumentProvider {
    override fun get(name: String) = provider.asSequence().map { it.get(name) }.filterNotNull().firstOrNull()
}

class OneVariableProvider(private val name: String) : ArgumentProvider {
    private var value = 0.0

    fun set(value: Double) {
        this.value = value
    }

    override fun get(name: String) = if (name == this.name) value else null
}

interface Function {
    fun eval(provider: ArgumentProvider): Double
    fun arity(): Int
    fun priority(): Int
    override fun toString(): String
    override fun equals(other: Any?): Boolean
    override fun hashCode(): Int
}

class Variable(private val name: String) : Function {
    override fun eval(provider: ArgumentProvider) =
        provider.get(name) ?: throw RuntimeException("No variable named $name")

    override fun arity() = 0
    override fun priority() = 10
    override fun toString() = name
    override fun equals(other: Any?) = (other is Variable) && name == other.name
    override fun hashCode() = name.hashCode()
}

class Const(val value: Double) : Function {
    override fun eval(provider: ArgumentProvider) = value
    override fun arity() = 0
    override fun priority() = 10
    override fun toString() = String.format("%.4f", this.value)
    override fun equals(other: Any?) = (other is Const) && FastMath.abs(value - other.value) < 0.1
    override fun hashCode() = value.hashCode()
}

class OneArgFunction(
    val type: Type,
    val arg: Function
) : Function {
    override fun eval(provider: ArgumentProvider) = type.fn(arg.eval(provider))
    override fun arity() = 1
    override fun priority() = 10
    override fun toString() = "${type.symbol}($arg)"
    override fun equals(other: Any?) = (other is OneArgFunction) && type == other.type && arg == other.arg
    override fun hashCode() = type.name.hashCode() xor arg.hashCode()

    enum class Type(
        val symbol: String,
        val fn: (Double) -> Double
    ) {
        ERF("erf", Erf::erf),
        SQRT("sqrt", FastMath::sqrt),
        EXP("exp", { x -> FastMath.min(FastMath.exp(x), 1E10) }),
        NORM("norm",  { x -> normImpl(x) });

        companion object {
            private val normImpl = object {
                private val dist = NormalDistribution()
                operator fun invoke(x: Double) = dist.density(x)
            }
        }
    }
}

class TwoArgFunction(
    val type: Type,
    val left: Function,
    val right: Function
) : Function {
    override fun eval(provider: ArgumentProvider) = type.fn(left.eval(provider), right.eval(provider))
    override fun arity() = 2
    override fun priority() = type.priority
    override fun toString(): String {
        val leftHigh = left.arity() >= 2 && left.priority() >= this.priority()
        val rightHigh = right.arity() >= 2 && right.priority() >= this.priority()

        return (if (leftHigh) "(" else "") + left.toString() + (if (leftHigh) ")" else "") + " " + type.symbol + " " +
                (if (rightHigh) "(" else "") + right.toString() + (if (rightHigh) ")" else "")
    }

    override fun equals(other: Any?) = (other is TwoArgFunction) && type == other.type && left == other.left && right == other.right
    override fun hashCode() = type.name.hashCode() xor left.hashCode() xor right.hashCode()

    enum class Type(
        val symbol: String,
        val fn: (Double, Double) -> Double,
        val priority: Int
    ) {
        ADD("+", { x, y -> x + y }, 0),
        SUB("-", { x, y -> x - y }, 0),
        MUL("*", { x, y -> x * y }, 1),
        DIV("/", { x, y -> x / y }, 1)
        //POW("^", { x, y -> Math.min(Math.pow(Math.abs(x) + EPS, y), 1 / EPS) }, 2)
    }
}

class NumericFunction(
    private val name: String,
    private val left: Double,
    private val leftValue: Double,
    private val right: Double,
    private val rightValue: Double,
    private val dx: Double,
    private val values: List<Double>,
    private val arg: Function
) : Function {
    override fun eval(provider: ArgumentProvider): Double {
        val x = arg.eval(provider)
        return when {
            x < left -> leftValue
            x > right -> rightValue
            else -> values[((x - left) / dx).toInt()]
        }
    }

    override fun arity() = 1
    override fun priority() = 10
    override fun toString() = "$name($arg)"
    override fun equals(other: Any?) = false
    override fun hashCode() = 0
}

class BoundFunction(
    private val original: Function,
    private val defaultProvider: ArgumentProvider
) : Function {
    override fun eval(provider: ArgumentProvider): Double {
        return original.eval(FallbackProvider(provider, defaultProvider))
    }

    override fun arity() = original.arity()
    override fun priority() = original.priority()
    override fun toString() = original.toString()
    override fun equals(other: Any?) = original == other
    override fun hashCode() = original.hashCode()
}

fun bind(function: Function, argumentProvider: ArgumentProvider): Function {
    return BoundFunction(function, argumentProvider);
}

fun integrate(
    name: String,
    function: Function,
    left: Double,
    right: Double,
    dx: Double,
    variable: String,
    provider: ArgumentProvider
): NumericFunction {
    val result = mutableListOf<Double>()
    var sum = 0.0
    val bound = bind(function, provider)
    val oneVariableProvider = OneVariableProvider(variable)
    var x = left
    while (x < right) {
        oneVariableProvider.set(x)
        sum += bound.eval(oneVariableProvider) * dx
        result.add(sum)
        x += dx
    }
    return NumericFunction(name, left, 0.0, right, sum, dx, result, Variable(variable))
}

fun product(
    f1: Function,
    f2: Function,
    left: Double,
    right: Double,
    dx: Double,
    variable: String,
    provider: ArgumentProvider
): Double {
    val b1 = bind(f1, provider)
    val b2 = bind(f2, provider)
    val oneVariableProvider = OneVariableProvider(variable)

    var mul = 0.0
    var sum1 = 0.0
    var sum2 = 0.0

    var x = left
    while (x < right) {
        oneVariableProvider.set(x)

        val v1 = b1.eval(oneVariableProvider)
        val v2 = b2.eval(oneVariableProvider)

        mul += v1 * v2
        sum1 += v1 * v1
        sum2 += v2 * v2

        x += dx
    }
    val EPS = 1E-5
    if (Math.abs(mul) < EPS && (Math.abs(sum1) < EPS || Math.abs(sum2) < EPS)) {
        return -1 / EPS
    }

    return mul / Math.sqrt(sum1 * sum2)
}

fun mse(
    f1: Function,
    f2: Function,
    left: Double,
    right: Double,
    dx: Double,
    variable: String,
    provider: ArgumentProvider
): Double {
    val b1 = bind(f1, provider)
    val b2 = bind(f2, provider)
    val oneVariableProvider = OneVariableProvider(variable)

    var sum = 0.0

    var x = left
    while (x < right) {
        oneVariableProvider.set(x)

        val v1 = b1.eval(oneVariableProvider)
        val v2 = b2.eval(oneVariableProvider)

        sum += FastMath.pow((v1 - v2) * (v1 - v2), 0.5)
        x += dx
    }

    return sum
}

operator fun Function.plus(value: Double): Function {
    return TwoArgFunction(TwoArgFunction.Type.ADD, this, Const(value))
}

operator fun Function.minus(value: Double): Function {
    return TwoArgFunction(TwoArgFunction.Type.SUB, this, Const(value))
}

operator fun Function.times(value: Double): Function {
    return TwoArgFunction(TwoArgFunction.Type.MUL, this, Const(value))
}

operator fun Function.div(value: Double): Function {
    return TwoArgFunction(TwoArgFunction.Type.DIV, this, Const(value))
}
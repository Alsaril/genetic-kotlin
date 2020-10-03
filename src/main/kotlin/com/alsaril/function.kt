package com.alsaril

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
    override fun toString(): String
}

class Variable(private val name: String) : Function {
    override fun eval(provider: ArgumentProvider) =
        provider.get(name) ?: throw RuntimeException("No variable named $name")

    override fun toString() = name
}

class Const(val value: Double) : Function {
    override fun eval(provider: ArgumentProvider) = value
    override fun toString() = value.toString()
}

class OneArgFunction(
    val name: String,
    val function: (Double) -> Double,
    val arg: Function
) : Function {
    override fun eval(provider: ArgumentProvider) = function(arg.eval(provider))
    override fun toString() = "$name($arg)"
}

class TwoArgFunction(
    val name: String,
    val function: (Double, Double) -> Double,
    val left: Function,
    val right: Function
) : Function {
    override fun eval(provider: ArgumentProvider) = function(left.eval(provider), right.eval(provider))
    override fun toString() = "$name($left, $right)"
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

    override fun toString() = "$name($arg)"
}

class BoundFunction(
    private val original: Function,
    private val defaultProvider: ArgumentProvider
) : Function {
    override fun eval(provider: ArgumentProvider): Double {
        return original.eval(FallbackProvider(provider, defaultProvider))
    }

    override fun toString() = original.toString()
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
        sum += bound.eval(oneVariableProvider)
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
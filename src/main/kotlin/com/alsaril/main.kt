package com.alsaril

import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.special.Erf
import org.apache.commons.math3.util.FastMath
import org.slf4j.LoggerFactory
import kotlin.math.max
import kotlin.system.exitProcess

class GeneticFunctionHelper(
    private val variable: String,
    private val initialDepth: Int,
    private val maxDepth: Int,
    private val mutationChance: Double,
    private val real: Function,
    private val left: Double,
    private val right: Double,
    private val dx: Double
) : GeneticHelper<Function> {
    private val LOG = LoggerFactory.getLogger(GeneticFunctionHelper::class.java)
    private val EPS = 1E-3

    private val constants = listOf(
        0.0, 1.0, -1.0, Math.PI, Math.E
    )

    private val oneArgFunctions = listOf<Pair<String, (Double) -> Double>>(
        //"sin" to FastMath::sin,
        //"cos" to FastMath::cos,
        "erf" to Erf::erf
        //"exp" to {x -> Math.min(Math.exp(x), 1 / EPS) },
    )

    private val twoArgFunctions = listOf<Triple<String, Int, (Double, Double) -> Double>>(
        Triple("+", 0, { x, y -> x + y }),
        Triple("-", 0, { x, y -> x - y }),
        Triple("*", 1, { x, y -> x * y }),
        Triple("/", 1, { x, y -> x * Math.signum(y) / (Math.abs(y) + EPS) })
        //"pow" to { x, y -> Math.min(Math.pow(Math.abs(x) + EPS, y), 1 / EPS) }
    )

    private fun leaf(f: Function) = f is Const || f is Variable

    private fun randomNode(depth: Int, random: Random): Function =
        when {
            depth == 0 -> when {   // generate leaf
                random.nextBoolean() -> Variable(variable)
                random.nextBoolean() -> Const(random.nextElement(constants))
                else -> Const((random.nextDouble() - 0.5) * 1000)
            }
            oneArgFunctions.isNotEmpty() && random.nextBoolean() -> { // one arg function
                val (name, fn) = random.nextElement(oneArgFunctions)
                OneArgFunction(name, fn, randomNode(depth - 1, random))
            }
            twoArgFunctions.isNotEmpty() -> { // two arg function
                val (name, priority, fn) = random.nextElement(twoArgFunctions)
                TwoArgFunction(name, fn, randomNode(depth - 1, random), randomNode(depth - 1, random), priority)
            }
            else -> throw IllegalStateException("wtf")
        }

    private fun _newInstance(random: Random) = randomNode(initialDepth, random)

    private fun _mutate(t: Function, random: Random): Function { // find random node and replace it with random
        if (random.nextDouble() < mutationChance) {
            return randomNode(initialDepth, random)
        }
        return when (t) {
            is Const -> {
                Const(t.value * (1 + random.nextDouble() / 5))
            }
            is Variable -> {
                if (random.nextDouble() < mutationChance) randomNode(initialDepth, random) else t
            }
            is OneArgFunction -> OneArgFunction(t.name, t.function, mutate(t.arg, random))
            is TwoArgFunction -> if (random.nextBoolean()) { // to the left
                TwoArgFunction(t.name, t.function, mutate(t.left, random), t.right, t.priority())
            } else { // to the right
                TwoArgFunction(t.name, t.function, t.left, mutate(t.right, random), t.priority())
            }
            else -> throw IllegalStateException("wtf")
        }
    }

    private fun _cross(t1: Function, t2: Function, random: Random): Function {
        if (t1 is Const && t2 is Const) {
            return Const((t1.value + t2.value) / 2)
        }
        if (leaf(t1) || leaf(t2)) {
            return if (random.nextBoolean()) t1 else t2
        }
        // now both have children

        // linear case
        if (t1 is OneArgFunction && t2 is OneArgFunction) {
            val (name, fn) = (if (random.nextBoolean()) t1.name to t1.function else t2.name to t2.function)
            return OneArgFunction(name, fn, cross(t1.arg, t2.arg, random))
        }

        // forked case
        if (t1 is TwoArgFunction && t2 is TwoArgFunction) {
            val (name, priority, fn) = (if (random.nextBoolean()) Triple(
                t1.name,
                t1.priority(),
                t1.function
            ) else Triple(t2.name, t2.priority(), t2.function))
            return TwoArgFunction(
                name,
                fn,
                cross(t1.left, t2.left, random),
                cross(t1.right, t2.right, random),
                priority
            )
        }

        // finally complicated case of different types
        val linear = (if (t1 is OneArgFunction) t1 else t2) as OneArgFunction
        val forked = (if (t1 is TwoArgFunction) t1 else t2) as TwoArgFunction

        return if (random.nextBoolean()) { // linear
            val forkedArg = if (random.nextBoolean()) forked.left else forked.right
            OneArgFunction(linear.name, linear.function, cross(linear.arg, forkedArg, random))
        } else { // forked
            val left = random.nextBoolean()
            TwoArgFunction(
                forked.name,
                forked.function,
                if (left) cross(linear.arg, forked.left, random) else forked.left,
                if (!left) cross(linear.arg, forked.right, random) else forked.right,
                forked.priority()
            )
        }
    }

    private fun protect(x: Double) = if (x.isNaN() || x.isInfinite()) 0.0 else x

    private fun optimize(f: Function, random: Random, maxDepth: Int): Function {
        if (leaf(f)) {
            return f
        }

        if (maxDepth == 1) {
            return randomNode(0, random)
        }

        if (f is OneArgFunction) {
            val arg = optimize(f.arg, random, maxDepth - 1)
            if (arg is Const) {
                return Const(protect(f.function(arg.value)))
            }
        }

        if (f is TwoArgFunction) {
            val left = optimize(f.left, random, maxDepth - 1)
            val right = optimize(f.right, random, maxDepth - 1)
            if (left is Const && right is Const) {
                return Const(f.function(protect(left.value), protect(right.value)))
            }

            if (left == right) {
                if (f.name == "-") return Const(0.0)
                if (f.name == "/") return Const(1.0)
            }

            if (left == Const(0.0)) {
                when (f.name) {
                    "+", "-" -> return right
                    "*", "/" -> return Const(0.0)
                }
            }

            if (right == Const(0.0)) {
                when (f.name) {
                    "+", "-" -> return right
                    "*" -> return Const(0.0)
                    "/" -> return Const(1 / EPS)
                }
            }
        }

        return f
    }

    private fun depth(t: Function): Int = when {
        leaf(t) -> 1
        t is OneArgFunction -> {
            depth(t.arg) + 1
        }
        t is TwoArgFunction -> {
            max(depth(t.left), depth(t.right)) + 1
        }
        else -> throw IllegalStateException("wtf")
    }

    override fun newInstance(random: Random) = optimize(_newInstance(random), random, maxDepth)
    override fun mutate(t: Function, random: Random) = optimize(_mutate(t, random), random, maxDepth)
    override fun cross(t1: Function, t2: Function, random: Random) = optimize(_cross(t1, t2, random), random, maxDepth)
    override fun score(t: Function) = -product(real, t, left, right, dx, variable, EmptyVariableProvider)
    override fun metrics(): List<Pair<String, (Function) -> Double>> = listOf("mse" to { fn -> mse(real, fn, left, right, dx, variable, EmptyVariableProvider)})
}

fun main() {
    val variable = "x"
    val left = -5.0
    val right = 5.0
    val dx = 1E-4

    val fn = object {
        private val dist = NormalDistribution()
        operator fun invoke(x: Double) = dist.density(x)
    }
    val real = integrate(
        "a",
        OneArgFunction("norm", { x -> fn(x) }, Variable(variable)),
        left,
        right,
        dx,
        variable,
        EmptyVariableProvider
    )
    val helper = GeneticFunctionHelper(variable, 4, 5, 0.1, real, left, right, dx)
    println(helper.score(OneArgFunction("sqr", { x -> Erf.erf(x / FastMath.sqrt(2.0)) / 2 }, Variable(variable))))
    val genetic = Genetic(helper, MultithreadedRandom(), 50, 10)
    genetic.train(100)
    val best = genetic.best().first
    println(best)

    exitProcess(0)
}
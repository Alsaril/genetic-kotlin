package com.alsaril

import org.slf4j.LoggerFactory
import java.util.*

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

    private val oneArgFunctions = listOf<Pair<String, (Double) -> Double>>(
        "sin" to Math::sin,
        "exp" to {x -> Math.min(Math.exp(x), 1 / EPS) }
    )

    private val twoArgFunctions = listOf<Pair<String, (Double, Double) -> Double>>(
        "+" to { x, y -> x + y },
        "-" to { x, y -> x - y },
        "*" to { x, y -> x * y },
        "/" to { x, y -> x * Math.signum(y) / (Math.abs(y) + EPS) },
        "pow" to { x, y -> Math.min(Math.pow(Math.abs(x) + EPS, y), 1 / EPS) }
    )

    private fun leaf(f: Function) = f is Const || f is Variable

    private fun randomNode(depth: Int, random: Random): Function =
        if (depth == 0) { // generate leaf
            if (random.nextBoolean()) { //variable
                Variable(variable)
            } else { // constant
                Const((random.nextDouble() - 0.5) * 1000) // (-500, 500)
            }
        } else if (random.nextBoolean()) { // one arg function
            val (name, fn) = random.nextRandom(oneArgFunctions)
            OneArgFunction(name, fn, randomNode(depth - 1, random))
        } else { // two arg function
            val (name, fn) = random.nextRandom(twoArgFunctions)
            TwoArgFunction(name, fn, randomNode(depth - 1, random), randomNode(depth - 1, random))
        }

    private fun _newInstance(random: Random) = randomNode(initialDepth, random)

    private fun _mutate(t: Function, random: Random): Function { // find random node and replace it with random
        if (random.nextDouble() < mutationChance) {
            return randomNode(initialDepth, random)
        }
        return when (t) {
            is Variable, is Const -> {
                if (random.nextDouble() < mutationChance) randomNode(initialDepth, random) else t
            }
            is OneArgFunction -> OneArgFunction(t.name, t.function, mutate(t.arg, random))
            is TwoArgFunction -> if (random.nextBoolean()) { // to the left
                TwoArgFunction(t.name, t.function, mutate(t.left, random), t.right)
            } else { // to the right
                TwoArgFunction(t.name, t.function, t.left, mutate(t.right, random))
            }
            else -> throw IllegalStateException("wtf")
        }
    }

    private fun _cross(t1: Function, t2: Function, random: Random): Function {
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
            val (name, fn) = (if (random.nextBoolean()) t1.name to t1.function else t2.name to t2.function)
            return TwoArgFunction(name, fn, cross(t1.left, t2.left, random), cross(t1.right, t2.right, random))
        }

        // finally complicated case of different types
        val linear = (if (t1 is OneArgFunction) t1 else t2) as OneArgFunction
        val forked = (if (t1 is TwoArgFunction) t1 else t2) as TwoArgFunction

        return if (random.nextBoolean()) { // linear
            val forkedArg = if (random.nextBoolean()) forked.left else forked.right
            OneArgFunction(linear.name, linear.function, cross(linear.arg, forkedArg, random))
        } else { // forked
            TwoArgFunction(
                forked.name,
                forked.function,
                cross(linear.arg, forked.left, random),
                cross(linear.arg, forked.right, random)
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
        }

        return f
    }

    override fun newInstance(random: Random) = optimize(_newInstance(random), random, maxDepth)
    override fun mutate(t: Function, random: Random) = optimize(_mutate(t, random), random, maxDepth)
    override fun cross(t1: Function, t2: Function, random: Random) = optimize(_cross(t1, t2, random), random, maxDepth)
    override fun score(t: Function) = product(real, t, left, right, dx, variable, EmptyVariableProvider)
}

fun <T> Random.nextRandom(list: List<T>): T = list[nextInt(list.size)]

fun main() {
    val variable = "x";
    val helper = GeneticFunctionHelper(variable, 2, 6, 0.001, OneArgFunction("sin", Math::sin, Variable("x")), 0.0, 100.0, 1E-3)
    val random = Random(42)
    val genetic = Genetic(helper, random, 50)
    genetic.train(10)
    val best = genetic.best()
    println(best)
}
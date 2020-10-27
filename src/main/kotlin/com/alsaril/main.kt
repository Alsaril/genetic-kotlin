package com.alsaril

import org.apache.commons.math3.util.FastMath
import java.awt.Color
import java.awt.Dimension
import java.awt.Graphics
import java.util.*
import javax.swing.JFrame
import javax.swing.JPanel
import kotlin.math.max

// function, variable, 20, left, right, dx
class GeneticFunctionHelper(
    private val derivative: Function,
    private val variable: String,
    private val variationCount: Int,
    private val left: Double,
    private val right: Double,
    private val dx: Double,
    private val random: Random,
    private val initialDepth: Int,
    private val maxDepth: Int,
    private val mutationChance: Double
) : GeneticHelper<Function> {
    // I'm using static variations, but they may be dynamic and potentially unstable through generations
    val variations: List<Pair<Function, ArgumentProvider>>
    private val vars = (derivative.vars() + variable).toList()
    private val parameters = derivative.vars() - setOf(variable)

    init {
        variations = (1..variationCount).map {
            val provider = MultiVariableProvider(parameters.map { v -> v to (random.nextDouble() - 0.5) * 3 })
            integrate("variation$it", derivative, left, right, dx, variable, provider) to provider
        }
    }

    private fun leaf(f: Function) = f is FixedConst || f is Const || f is Variable

    private fun randomNode(depth: Int, random: Random): Function =
        when {
            depth == 0 -> when (random.nextInt(3)) {   // generate leaf
                0 -> Variable(random.nextElement(vars))
                1 -> FixedConst(random.nextElement(FixedConst.Type.values()))
                2 -> Const(random.nextInt(1000) - 500)
                else -> throw IllegalStateException("wtf")
            }
            random.nextBoolean() -> { // one arg function
                val type = random.nextElement(UnaryFunction.Type.values())
                UnaryFunction(type, randomNode(depth - 1, random))
            }
            // two arg function
            else -> {
                val type = random.nextElement(BinaryFunction.Type.values())
                BinaryFunction(type, randomNode(depth - 1, random), randomNode(depth - 1, random))
            }
        }

    private fun _newInstance(random: Random) = randomNode(initialDepth, random)

    private fun _mutate(t: Function, random: Random): Function { // find random node and replace it with random
        if (random.nextDouble() < mutationChance) {
            return when (random.nextInt(5)) {
                0 -> BinaryFunction(
                    BinaryFunction.Type.MUL,
                    Const(random.nextInt(10) + 1),
                    t
                ) // multiply subtree with k around 1
                1 -> BinaryFunction(
                    BinaryFunction.Type.ADD,
                    Const(random.nextInt(10) - 5),
                    t
                ) // add to subtree shift around 0
                2 -> randomNode(initialDepth, random) // generate brand new subtree
                3 -> when (t) { // remove node from tree
                    is Const, is Variable, is FixedConst -> t
                    is UnaryFunction -> t.arg
                    is BinaryFunction -> if (random.nextBoolean()) t.left else t.right
                    else -> throw IllegalStateException("wtf")
                }
                4 -> if (random.nextBoolean()) { // add one argument node
                    val type = random.nextElement(UnaryFunction.Type.values())
                    UnaryFunction(type, t)
                } else { // add two argument node
                    val type = random.nextElement(BinaryFunction.Type.values())
                    if (random.nextBoolean()) {
                        BinaryFunction(type, t, randomNode(initialDepth, random))
                    } else {
                        BinaryFunction(type, randomNode(initialDepth, random), t)
                    }
                }
                else -> throw IllegalStateException("wtf")
            }
        }
        return when (t) {
            is Const, is Variable, is FixedConst -> t
            is UnaryFunction -> UnaryFunction(t.type, mutate(t.arg, random))
            is BinaryFunction -> if (random.nextBoolean()) {
                BinaryFunction(t.type, mutate(t.left, random), t.right)
            } else {
                BinaryFunction(t.type, t.right, mutate(t.right, random))
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
        if (t1 is UnaryFunction && t2 is UnaryFunction) {
            val type = (if (random.nextBoolean()) t1.type else t2.type)
            return UnaryFunction(type, cross(t1.arg, t2.arg, random))
        }

        // forked case
        if (t1 is BinaryFunction && t2 is BinaryFunction) {
            val type = if (random.nextBoolean()) t1.type else t2.type
            return BinaryFunction(type, cross(t1.left, t2.left, random), cross(t1.right, t2.right, random))
        }

        // finally complicated case of different types
        val linear = (if (t1 is UnaryFunction) t1 else t2) as UnaryFunction
        val forked = (if (t1 is BinaryFunction) t1 else t2) as BinaryFunction

        return if (random.nextBoolean()) { // linear
            val forkedArg = if (random.nextBoolean()) forked.left else forked.right
            UnaryFunction(linear.type, cross(linear.arg, forkedArg, random))
        } else { // forked
            val left = random.nextBoolean()
            BinaryFunction(
                forked.type,
                if (left) cross(linear.arg, forked.left, random) else forked.left,
                if (!left) cross(linear.arg, forked.right, random) else forked.right
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

        if (f is UnaryFunction) {
            val arg = optimize(f.arg, random, maxDepth - 1)
            if (arg is Const) {
                val optimized = f.type.fn(arg.value.toDouble())
                if (FastMath.abs(optimized - optimized.toInt()) < 1E-5) return Const(optimized.toInt())
            }
            return if (f.arg === arg) f else UnaryFunction(f.type, arg)
        }

        if (f is BinaryFunction) {
            val left = optimize(f.left, random, maxDepth - 1)
            val right = optimize(f.right, random, maxDepth - 1)
            if (left is Const && right is Const) {
                val optimized = f.type.fn(left.value.toDouble(), right.value.toDouble())
                if (FastMath.abs(optimized - optimized.toInt()) < 1E-5) return Const(optimized.toInt())
            }

            if (left == right) {
                when (f.type) {
                    BinaryFunction.Type.SUB -> return Const(0)
                    BinaryFunction.Type.DIV -> return Const(1)
                }
            }

            if (left == Const(0)) {
                 when (f.type) {
                     BinaryFunction.Type.ADD, BinaryFunction.Type.SUB -> return right
                     BinaryFunction.Type.MUL, BinaryFunction.Type.DIV -> return Const(0)
                    //TwoArgFunction.Type.POW -> Const(0)
                }
            }

           // if (left == Const(1) && f.type == TwoArgFunction.Type.POW) return Const(1)

            if (right == Const(0)) {
                 when (f.type) {
                     BinaryFunction.Type.ADD, BinaryFunction.Type.SUB -> return right
                     BinaryFunction.Type.MUL -> return Const(0)
                     BinaryFunction.Type.DIV -> return Const(100000)
                    //TwoArgFunction.Type.POW -> Const(1)
                }
            }

            if (f.type == BinaryFunction.Type.SUB && right is Const && right.value < 0) {
                return BinaryFunction(BinaryFunction.Type.ADD, f.left, Const(-right.value))
            }

            return if (f.left === left && f.right === right) f else BinaryFunction(f.type, left, right)
        }

        throw IllegalStateException("wtf")
    }

    private fun depth(t: Function): Int = when {
        leaf(t) -> 1
        t is UnaryFunction -> {
            depth(t.arg) + 1
        }
        t is BinaryFunction -> {
            max(depth(t.left), depth(t.right)) + 1
        }
        else -> throw IllegalStateException("wtf")
    }

    override fun newInstance(random: Random) = optimize(_newInstance(random), random, maxDepth)
    override fun mutate(t: Function, random: Random) = optimize(_mutate(t, random), random, maxDepth)
    override fun cross(t1: Function, t2: Function, random: Random) = optimize(_cross(t1, t2, random), random, maxDepth)
    override fun score(t: Function) = variations.asSequence()
        .map { (f, p) -> mse(f, t, left, right, dx, variable, p) }.sum()

    override fun metrics(): List<Pair<String, (Function) -> Double>> = emptyList()
    //listOf("product" to { fn -> product(real, fn, left, right, dx, variable, EmptyVariableProvider) })
}

class Plotter(width: Int, height: Int, private val scale: Double, private val Y: Int) {
    private val colors = listOf<Color>(Color.ORANGE, Color(0, 180, 50), Color.CYAN)

    private val frame = JFrame()
    private val panel = object : JPanel() {
        override fun paint(g: Graphics) {
            g.color = Color.DARK_GRAY
            g.fillRect(0, 0, width, height)
            objects.forEachIndexed { i, f ->
                g.color = colors[i]
                val left = -width / 2.0 * scale
                val right = width / 2.0 * scale
                val dx = scale
                var x = left
                val provider = OneVariableProvider("x")
                provider.set(x)
                var xo = left
                var yo = f.eval(provider)
                x += dx
                while (x < right) {
                    provider.set(x)
                    val xn = x
                    val yn = f.eval(provider)
                    g.drawLine(
                        (xo / scale).toInt() + width / 2,
                        -(yo / scale).toInt() + height / 2 + Y,
                        (xn / scale).toInt() + width / 2,
                        -(yn / scale).toInt() + height / 2 + Y
                    )
                    x += dx
                    xo = xn
                    yo = yn
                }
            }
        }
    }
    private val objects = Collections.synchronizedList(mutableListOf<Function>())

    init {
        panel.preferredSize = Dimension(width, height)
        frame.add(panel)
        frame.pack()
        frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
        frame.isVisible = true
    }

    fun plot(title: Any, vararg functions: Function) {
        objects.clear()
        objects.addAll(functions)
        panel.repaint()
        frame.title = title.toString()
    }
}

fun main() {
    val variable = "x"
    val function = object : Function {
        fun density(x: Double, mean: Double, sigma: Double): Double {
            return FastMath.exp(logDensity(x, mean, sigma))
        }

        fun logDensity(x: Double, mean: Double, sigma: Double): Double {
            val x0: Double = x - mean
            val x1: Double = x0 / sigma
            return -0.5 * x1 * x1 - FastMath.log(sigma) + 0.5 * FastMath.log(2 * FastMath.PI)
        }

        override fun eval(provider: ArgumentProvider): Double {
            val x = provider.get(variable)!!
            val m = provider.get("m")!!
            val s = FastMath.abs(provider.get("s")!!)
            return density(x, m, s)
        }

        override fun arity() = 3
        override fun priority() = 100
        override fun vars() = setOf(variable, "m", "s")
        override fun toString() = "None"
        override fun equals(other: Any?) = false
        override fun hashCode() = 0
    }
    val left = -3.0
    val right = 3.0
    val dx = 1E-3

    val random = MultithreadedRandom()

    val parameters = function.vars() - setOf(variable)
    val variations = (1..5).map {
        val provider = MultiVariableProvider(parameters.map { v -> v to (random.nextDouble() - 0.5) * 3 })
        integrate("variation$it", function, left, right, dx, variable, provider) to provider
    }

    val generator = functionGenerator(function.vars())
    var i = 0
    generator
        .filter { it.vars().contains("x") }
        .mapParallel(100) {
            if (i % 10000 == 0) println("$i: $it");
            it to variations.asSequence().map { (f, p) -> mse(f, it, left, right, dx, variable, p) }.sum()
        }.minify { it.second }
        .forEach { if (i++ % 10000 == 0) println(it) }

//
//    val random = MultithreadedRandom()
//    val helper = GeneticFunctionHelper(function, name, 10, left, right, dx, random, 3, 6, 0.3)
//    val genetic = Genetic(helper, random, 20, 20, Const(0))
//    val plotter = Plotter(800, 800, 0.02,  0)
//    genetic.train(10000) { (epoch, function) ->
//        val index = epoch / 10 % 10
//        val vari = helper.variations[index]
//        val bound = bind(function, vari.second)
//        plotter.plot(index, vari.first, bound)
//    }
//    val best = genetic.best().first
//    println(best)
}
package com.alsaril

import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.special.Erf
import org.apache.commons.math3.util.FastMath
import org.slf4j.LoggerFactory
import java.awt.Color
import java.awt.Dimension
import java.awt.Graphics
import java.util.*
import javax.swing.JFrame
import javax.swing.JPanel
import kotlin.math.max

private val constants = listOf(
    0.0, 1.0, -1.0, Math.PI, Math.E
)

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

    private fun leaf(f: Function) = f is Const || f is Variable

    private fun randomNode(depth: Int, random: Random): Function =
        when {
            depth == 0 -> when {   // generate leaf
                random.nextBoolean() -> Variable(variable)
                random.nextBoolean() -> Const(random.nextElement(constants))
                else -> Const((random.nextDouble() - 0.5) * 1000)
            }
            random.nextBoolean() -> { // one arg function
                val type = random.nextElement(OneArgFunction.Type.values())
                OneArgFunction(type, randomNode(depth - 1, random))
            }
            // two arg function
            else -> {
                val type = random.nextElement(TwoArgFunction.Type.values())
                TwoArgFunction(type, randomNode(depth - 1, random), randomNode(depth - 1, random))
            }
        }

    private fun _newInstance(random: Random) = randomNode(initialDepth, random)

    private fun _mutate(t: Function, random: Random): Function { // find random node and replace it with random
        if (leaf(t)) return t
        if (random.nextDouble() < mutationChance) {
           return when (random.nextInt(5)) {
                0 -> TwoArgFunction(TwoArgFunction.Type.MUL, Const(1 + (random.nextDouble() - 0.5) / 5), t) // multiply subtree with k around 1
                1 -> TwoArgFunction(TwoArgFunction.Type.ADD, Const((random.nextDouble() - 0.5) / 5), t) // add to subtree shift around 0
                2 -> randomNode(initialDepth, random) // generate brand new subtree
                3 -> when (t) { // remove node from tree
                    is OneArgFunction -> t.arg
                    is TwoArgFunction -> if (random.nextBoolean()) t.left else t.right
                    else -> throw IllegalStateException("wtf")
                }
                4 -> if (random.nextBoolean()) { // add one argument node
                    val type = random.nextElement(OneArgFunction.Type.values())
                    OneArgFunction(type, t)
                } else { // add two argument node
                    val type = random.nextElement(TwoArgFunction.Type.values())
                    val left = random.nextBoolean()
                    TwoArgFunction(type, if (left) t else randomNode(initialDepth, random), if (!left) t else randomNode(initialDepth, random))
                }
                else -> throw IllegalStateException("wtf")
            }
        }
        return when (t) {
            is OneArgFunction -> OneArgFunction(t.type, mutate(t.arg, random))
            is TwoArgFunction -> if (random.nextBoolean()) {
                TwoArgFunction(t.type, mutate(t.left, random), t.right)
            } else {
                TwoArgFunction(t.type, t.right, mutate(t.right, random))
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
            val type = (if (random.nextBoolean()) t1.type else t2.type)
            return OneArgFunction(type, cross(t1.arg, t2.arg, random))
        }

        // forked case
        if (t1 is TwoArgFunction && t2 is TwoArgFunction) {
            val type = if (random.nextBoolean()) t1.type else t2.type
            return TwoArgFunction(type, cross(t1.left, t2.left, random), cross(t1.right, t2.right, random))
        }

        // finally complicated case of different types
        val linear = (if (t1 is OneArgFunction) t1 else t2) as OneArgFunction
        val forked = (if (t1 is TwoArgFunction) t1 else t2) as TwoArgFunction

        return if (random.nextBoolean()) { // linear
            val forkedArg = if (random.nextBoolean()) forked.left else forked.right
            OneArgFunction(linear.type, cross(linear.arg, forkedArg, random))
        } else { // forked
            val left = random.nextBoolean()
            TwoArgFunction(
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

        if (f is OneArgFunction) {
            val arg = optimize(f.arg, random, maxDepth - 1)
            if (arg is Const) {
                return Const(protect(f.type.fn(arg.value)))
            }
            return if (f.arg === arg) f else OneArgFunction(f.type, arg)
        }

        if (f is TwoArgFunction) {
            val left = optimize(f.left, random, maxDepth - 1)
            val right = optimize(f.right, random, maxDepth - 1)
            if (left is Const && right is Const) {
                return Const(f.type.fn(protect(left.value), protect(right.value)))
            }

            if (left == right) {
                when (f.type) {
                    TwoArgFunction.Type.SUB -> return Const(0.0)
                    TwoArgFunction.Type.DIV -> return Const(1.0)
                }
            }

            if (left == Const(0.0)) {
                return when (f.type) {
                    TwoArgFunction.Type.ADD, TwoArgFunction.Type.SUB -> right
                    TwoArgFunction.Type.MUL, TwoArgFunction.Type.DIV -> Const(0.0)
                }
            }

            if (right == Const(0.0)) {
                return when (f.type) {
                    TwoArgFunction.Type.ADD, TwoArgFunction.Type.SUB -> right
                    TwoArgFunction.Type.MUL -> Const(0.0)
                    TwoArgFunction.Type.DIV -> Const(1 / EPS)
                }
            }

            if (f.type == TwoArgFunction.Type.SUB && right is Const && right.value < 0) {
                return TwoArgFunction(TwoArgFunction.Type.ADD, f.left, Const(-right.value))
            }

            return if (f.left === left && f.right === right) f else TwoArgFunction(f.type, left, right)
        }

        throw IllegalStateException("wtf")
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
    override fun score(t: Function) = mse(real, t, left, right, dx, variable, EmptyVariableProvider)
    override fun metrics(): List<Pair<String, (Function) -> Double>> =
        listOf("product" to { fn -> product(real, fn, left, right, dx, variable, EmptyVariableProvider) })
}

class Plotter(width: Int, height: Int, private val scale: Double) {
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
                        -(yo / scale).toInt() + height / 2,
                        (xn / scale).toInt() + width / 2,
                        -(yn / scale).toInt() + height / 2
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

    fun plot(vararg functions: Function) {
        objects.clear()
        objects.addAll(functions)
        panel.repaint()
    }
}

fun answer(variable: String): Function {
    return OneArgFunction(OneArgFunction.Type.ERF, Variable(variable) / FastMath.sqrt(2.0)) / 2.0 + 0.5
}

fun main() {
    val variable = "x"
    val left = -3.0
    val right = 3.0
    val dx = 5E-4


    val real = integrate(
        "a",
        OneArgFunction(OneArgFunction.Type.NORM, Variable(variable)),
        left,
        right,
        dx,
        variable,
        EmptyVariableProvider
    )
    val helper = GeneticFunctionHelper(variable, 4, 6, 0.3, real, left, right, dx)
    println(helper.score(answer(variable)))
    val genetic = Genetic(helper, MultithreadedRandom(), 20, 100, Const(1.0))
    val plotter = Plotter(800, 400, .01)
    genetic.train(10000) {
        plotter.plot(real, it)
    }
    val best = genetic.best().first
    println(best)
}
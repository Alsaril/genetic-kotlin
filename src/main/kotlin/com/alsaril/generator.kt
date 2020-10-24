package com.alsaril

interface Generator<T> {
    fun hasNext(): Boolean
    fun next(): T
    fun reset()
}

object EmptyGenerator : Generator<Any> {
    override fun hasNext() = false
    override fun next() = throw NoSuchElementException()
    override fun reset() {}
}

class ListGenerator<T>(private val list: List<T>): Generator<T> {
    private var pos = 0
    override fun hasNext() = pos != list.size
    override fun next() = list[pos++]
    override fun reset() {
        pos = 0
    }
}

class SequenceGenerator : Generator<Int> {
    private var value = 0
    override fun hasNext() = true
    override fun next() = value++
    override fun reset() {
        value = 0
    }
}

@Suppress("UNCHECKED_CAST")
fun <T> emptyGenerator() = EmptyGenerator as Generator<T>

fun <T> Generator<T>.take(limit: Int) = object : Generator<T> {
    private var returned = 0

    override fun hasNext() = returned < limit && this@take.hasNext()

    override fun next(): T {
        returned++
        return this@take.next()
    }

    override fun reset() {
        returned = 0
        this@take.reset()
    }
}

fun <T> Generator<T>.forEach(action: (T) -> Unit) {
    while (this.hasNext()) {
        action(this.next())
    }
}

fun <T> Generator<T>.toList(): List<T> {
    val result = mutableListOf<T>()
    while (this.hasNext()) {
        result.add(this.next())
    }
    return result
}

fun <A, B> Generator<A>.map(transform: (A) -> B) = object : Generator<B> {
    override fun hasNext() = this@map.hasNext()
    override fun next() = transform(this@map.next())
    override fun reset() = this@map.reset()
}

fun <A, B> Generator<A>.flatMap(transform: (A) -> Generator<B>) = object : Generator<B> {
    private var currentGenerator: Generator<B>? = null
    private var advanced = false
    private var hasNext = false
    private var next: B? = null

    fun tryAdvance() {
        if (advanced) return

        var found = currentGenerator?.hasNext() ?: false
        while ((currentGenerator == null || !currentGenerator!!.hasNext()) && this@flatMap.hasNext()) {
            val el = transform(this@flatMap.next())
            if (el.hasNext()) { // have found nonempty element
                found = true
                currentGenerator = el
                break
            }
        }

        hasNext = found
        if (found) {
            next = currentGenerator!!.next()
        }
        advanced = true
    }

    override fun hasNext(): Boolean {
        tryAdvance()
        return hasNext
    }

    override fun next(): B {
        tryAdvance()
        if (hasNext) {
            advanced = false
            return next!!
        } else {
            throw NoSuchElementException()
        }
    }

    override fun reset() = this@flatMap.reset()
}

infix fun <T> Generator<T>.then(that: Generator<T>) = SequenceGenerator().take(2).flatMap { if (it == 0) this@then else that }

// left and right will be changed in process!
fun <L, R> makePair(left: Generator<L>, right: Generator<R>) = right.flatMap { rv ->
    left.reset()
    left.map { lv -> lv to rv }
}

fun functionGenerator(): Generator<Function> {
    return SequenceGenerator().flatMap { generateLevel(it) }
}

fun generateLevel(level: Int): Generator<Function> {
    if (level == 0) {
        return leafGenerator()
    }
    return unaryGenerator(level) then binaryGenerator(level)
}

val leafFunctions: List<Function> = FixedConst.Type.values().map { FixedConst(it) } + (-10 until 10).map { Const(it) }
fun leafGenerator() = ListGenerator(leafFunctions)

val unaryTypes = UnaryFunction.Type.values().toList()
fun unaryTypesGenerator() = ListGenerator(unaryTypes)

fun unaryGenerator(level: Int): Generator<Function> {
    return unaryTypesGenerator().flatMap { type -> generateLevel(level - 1).map { UnaryFunction(type, it) as Function } }
}

val binaryTypes = BinaryFunction.Type.values().toList()
fun binaryTypesGenerator() = ListGenerator(binaryTypes)

fun binaryLevelGenerator(level: Int): Generator<Pair<Int, Int>> {
    val list = mutableListOf<Pair<Int, Int>>()
    repeat(level) {
        list.add(it to level - 1)
        list.add(level - 1 to it)
    }
    list.removeAt(list.lastIndex)
    return ListGenerator(list)
}

fun binaryGenerator(level: Int): Generator<Function> {
    return binaryTypesGenerator().flatMap { type ->
        binaryLevelGenerator(level).flatMap { (leftLevel, rightLevel) -> makePair(generateLevel(leftLevel), generateLevel(rightLevel)).map { BinaryFunction(type, it.first, it.second) as Function } }
    }
}
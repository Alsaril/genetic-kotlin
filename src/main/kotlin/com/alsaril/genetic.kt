package com.alsaril

import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.Future

interface Random {
    fun nextInt(): Int
    fun nextInt(bound: Int): Int
    fun nextDouble(): Double
    fun nextBoolean(): Boolean
    fun <T> nextElement(list: List<T>) = list[nextInt(list.size)]
    fun <T> nextElement(arr: Array<T>) = arr[nextInt(arr.size)]
}

class MultithreadedRandom : Random {
    private val localRandom = ThreadLocal.withInitial { java.util.Random() }

    override fun nextInt() = localRandom.get().nextInt()
    override fun nextInt(bound: Int) = localRandom.get().nextInt(bound)
    override fun nextDouble() = localRandom.get().nextDouble()
    override fun nextBoolean() = localRandom.get().nextBoolean()
}

interface GeneticHelper<T> {
    fun newInstance(random: Random): T
    fun mutate(t: T, random: Random): T
    fun cross(t1: T, t2: T, random: Random): T
    fun score(t: T): Double
    fun metrics(): List<Pair<String, (T) -> Double>>
}

class Genetic<T>(
    private val geneticHelper: GeneticHelper<T>,
    private val random: Random,
    private val size: Int,
    threads: Int,
    init: T
) {
    private val pool = Executors.newFixedThreadPool(threads)
    private var population = mutableListOf<Pair<T, Double>>()

    init {
        (1 until size).map {
            createWithScore(geneticHelper.newInstance(random))
        }.forEach { population.add(it.get()) }
        createWithScore(init).get()?.let { population.add(it) }
        population.sortBy { (e, s) -> s }
    }

    private fun createWithScore(t: T): Future<Pair<T, Double>> =
        pool.submit(Callable<Pair<T, Double>> { t to geneticHelper.score(t) })

    fun best() = population.first()

    fun train(epoch: Int, callback: ((T) -> Unit)? = null) {
        repeat(epoch) {
            val start = System.currentTimeMillis()
            val newPopulation = mutableListOf<Pair<T, Double>>()
            val futures = mutableListOf<Future<Pair<T, Double>>>()

            // copy first 5
            repeat(5) {
                createWithScore(population[it].first).let(futures::add)
            }

            repeat(5) {
                createWithScore(geneticHelper.cross(population[it].first, population[random.nextInt(size)].first, random)).let(futures::add)
            }

            // mutate all
            repeat(size) {
                createWithScore(geneticHelper.mutate(population[it].first, random)).let(futures::add)
            }

            // new 10
            repeat(10) {
                createWithScore(geneticHelper.newInstance(random)).let(futures::add)
            }

            // cross first 5 with random
            repeat(5) {
                createWithScore(
                    geneticHelper.cross(population[it].first, population[random.nextInt(size)].first, random)
                ).let(futures::add)
            }

            // 50 random crosses
            repeat(50) {
                createWithScore(
                    geneticHelper.cross(
                        population[random.nextInt(size)].first,
                        population[random.nextInt(population.size)].first,
                        random
                    )
                ).let(futures::add)
            }

            futures.forEach { newPopulation.add(it.get()) }

            val st = mutableSetOf<T>()
            val realNew = mutableListOf<Pair<T, Double>>()
            newPopulation.forEach { (t, score) ->
                if (t !in st) {
                    st.add(t); realNew.add(t to score)
                }
            }

            if (realNew.size < size) {
                val newFutures = mutableListOf<Future<Pair<T, Double>>>()
                repeat(size - realNew.size) {
                    createWithScore(geneticHelper.mutate(population[it].first, random)).let(newFutures::add)
                }
                newFutures.forEach { realNew.add(it.get()) }
                require(realNew.size == size)
                population = realNew
                population.sortBy { (e, s) -> s }
            } else {
                realNew.sortBy { (e, s) -> s }
                population = realNew.subList(0, size)
            }

            require(population.size == size)

            val end = System.currentTimeMillis()

            print("Epoch ${String.format("%3d", it)}:    time  ${String.format("%3d", end - start)} ms,    score  ${String.format("%.2f", geneticHelper.score(population[0].first))}, ")
            geneticHelper.metrics().forEach { (name, fn) -> print("    $name  ${String.format("%.4f", fn(population[0].first))}") }
            println(",    best  ${population.first().first}")

            callback?.invoke(population.first().first)
        }
    }

}
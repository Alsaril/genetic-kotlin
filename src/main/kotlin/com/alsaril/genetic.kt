package com.alsaril

import java.util.*
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.Future

interface Random {
    fun nextInt(): Int
    fun nextInt(bound: Int): Int
    fun nextDouble(): Double
    fun nextBoolean(): Boolean
    fun <T> nextElement(list: List<T>) = list[nextInt(list.size)]
}

class MultithreadedRandom: Random {
    private val localRandom = ThreadLocal.withInitial { Random() }

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
    threads: Int
) {
    private val pool = Executors.newFixedThreadPool(threads)
    private var population = mutableListOf<Pair<T, Double>>()

    init {
        (1..size).map {
            createWithScore(geneticHelper.newInstance(random))
        }.forEach { population.add(it.get()) }
        population.sortBy { (e, s) -> s }
    }

    private fun createWithScore(t: T): Future<Pair<T, Double>> = pool.submit(Callable<Pair<T, Double>> { t to geneticHelper.score(t) })

    fun best() = population.first()

    fun train(epoch: Int) {
        repeat(epoch) {
            val start = System.currentTimeMillis()
            val newPopulation = Collections.synchronizedList(mutableListOf<Pair<T, Double>>())
            val futures = mutableListOf<Future<Pair<T, Double>>>()

            repeat(5) {
                createWithScore(geneticHelper.mutate(population[it].first, random)).let(futures::add)
            }

            repeat(9) {
                createWithScore(geneticHelper.newInstance(random)).let(futures::add)
            }

            repeat(5) {
                createWithScore(
                    geneticHelper.cross(population[it].first, population[random.nextInt(size)].first, random)
                ).let(futures::add)
            }

            repeat(size - 20) {
                createWithScore(
                    geneticHelper.cross(
                        population[random.nextInt(size)].first,
                        population[random.nextInt(population.size)].first,
                        random
                    )
                ).let(futures::add)
            }

            repeat(1) {
                createWithScore(population[it].first).let(futures::add)
            }

            futures.forEach { newPopulation.add(it.get()) }

            require(newPopulation.size == size)
            population = newPopulation
            population.sortBy { (e, s) -> s }

            val end = System.currentTimeMillis()

            print("Epoch $it: time ${end - start} ms, score ${geneticHelper.score(population[0].first)}, ")
            geneticHelper.metrics().forEach { (name, fn) -> print("$name ${fn(population[0].first)}") }
            println(", best\t${population.first().first}")
        }
    }

}
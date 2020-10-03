package com.alsaril

import java.util.*

interface GeneticHelper<T> {
    fun newInstance(random: Random): T
    fun mutate(t: T, random: Random): T
    fun cross(t1: T, t2: T, random: Random): T
    fun score(t: T): Double
}

class Genetic<T>(
    private val geneticHelper: GeneticHelper<T>,
    private val random: Random,
    private val size: Int
) {
    private var population = mutableListOf<Pair<T, Double>>()

    init {
        repeat(size) {
            addWithScore(population, geneticHelper.newInstance(random))
        }
        population.sortByDescending { (e, s) -> s }
    }

    private fun addWithScore(list: MutableList<Pair<T, Double>>, t: T) {
        val score = geneticHelper.score(t)
        list.add(t to score)
    }

    fun best() = population.first()

    fun train(epoch: Int) {
        repeat(epoch) {
            val start = System.currentTimeMillis()
            val newPopulation = mutableListOf<Pair<T, Double>>()

            repeat(5) {
                addWithScore(newPopulation, geneticHelper.mutate(population[it].first, random))
            }

            repeat(5) {
                addWithScore(
                    newPopulation,
                    geneticHelper.cross(population[it].first, population[random.nextInt(size)].first, random)
                )
            }

            repeat(size - newPopulation.size - 5) {
                addWithScore(
                    newPopulation,
                    geneticHelper.cross(
                        population[random.nextInt(size)].first,
                        population[random.nextInt(population.size)].first,
                        random
                    )
                )
            }

            repeat(5) {
                addWithScore(newPopulation,population[it].first)
             }

            require(newPopulation.size == size)
            population = newPopulation
            population.sortByDescending { (e, s) -> s }

            val end = System.currentTimeMillis()

            println("Epoch $it: time ${end - start} ms, score ${geneticHelper.score(population[0].first)}")
        }
    }

}
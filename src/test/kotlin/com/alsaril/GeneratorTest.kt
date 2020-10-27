package com.alsaril

import org.junit.Assert.assertEquals
import org.junit.Test

class GeneratorTest {
    @Test
    fun sequenceTest() {
        assertEquals((0 until 10).toList(), SequenceGenerator().take(10).toList())
    }

    @Test
    fun skipTest() {
        assertEquals((5 until 15).toList(), SequenceGenerator().skip(5).take(10).toList())
    }

    @Test
    fun mapTest() {
        assertEquals((0 until 10).map { it * 2 }.toList(), SequenceGenerator().take(10).map { it * 2 }.toList())
    }

    @Test
    fun filterTest() {
        assertEquals((0 until 10).filter { it % 3 == 0 }.toList(), SequenceGenerator().take(10).filter { it % 3 == 0 }.toList())
    }

    @Test
    fun flatMapTest() {
        assertEquals(
            (0 until 10).flatMap { 0 until it }.toList(),
            SequenceGenerator().take(10).flatMap { SequenceGenerator().take(it) }.toList()
        )
    }

    @Test
    fun thenTest() {
        val first = SequenceGenerator().take(5)
        val second = SequenceGenerator().take(5)
        assertEquals((0 until 10).map { it % 5 },  (first then second).toList())
    }

    @Test
    fun makePairTest() {
        val left = SequenceGenerator().take(3)
        val right = SequenceGenerator().take(3)
        val real = mutableListOf<Pair<Int, Int>>()
        repeat(3) { i ->
            repeat(3) { j ->
                real.add(j to i)
            }
        }
        assertEquals(real, makePair(left, right).toList())
    }

    @Test
    fun binaryLevelGeneratorTest() {
        assertEquals(listOf(0 to 0), binaryLevelGenerator(1).toList())
        assertEquals(listOf(0 to 1, 1 to 0, 1 to 1), binaryLevelGenerator(2).toList())
        assertEquals(listOf(0 to 2, 2 to 0, 1 to 2, 2 to 1, 2 to 2), binaryLevelGenerator(3).toList())
    }
}
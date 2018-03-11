"""
A game of life implementation in TensorFlow.

Averages 8ns / cell on my rather slow laptop.
"""

import sys
import time

import numpy as np
import tensorflow as tf

def main():
    """
    Run a few examples.
    """
    run_icolumn_demo()
    run_benchmark()

def run_icolumn_demo():
    """
    Print the cycles through an I-column.
    """
    icolumn = np.zeros((18, 11), dtype='bool')
    create_icolumn(icolumn, 0, 0)
    board = tf.Variable(icolumn)
    step = tf.assign(board, update_board(board))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print_board(sess.run(board))
        print('-' * 11)
        for _ in range(16):
            print_board(sess.run(step))
            print('-' * 11)

def run_benchmark():
    """
    Benchmark the stepping performance.
    """
    init = np.zeros((1000, 1000), dtype='bool')
    for i in range(0, 1000 // 18):
        for j in range(0, 1000 // 18):
            create_icolumn(init, i, j)
    board = tf.Variable(init)
    step = tf.assign(board, update_board(board))
    print('Running benchmark...')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start = time.time()
        for _ in range(10):
            sess.run(step)
        print('ns per cell: %f' % (1e9 * (time.time() - start) / (1000 * 1000 * 10)))

def count_neighbors(board):
    """
    Turn a 2-D bool Tensor into a 2-D uint8 Tensor
    counting the active neighbor cells.
    """
    def _shift_down(board):
        return tf.concat([board[-1:], board[:-1]], axis=0)
    def _shift_up(board):
        return tf.concat([board[1:], board[:1]], axis=0)
    def _shift_right(board):
        return tf.concat([board[:, -1:], board[:, :-1]], axis=1)
    def _shift_left(board):
        return tf.concat([board[:, 1:], board[:, :1]], axis=1)
    int_board = tf.cast(board, tf.uint8)
    down = _shift_down(int_board)
    up = _shift_up(int_board)
    right = _shift_right(int_board)
    left = _shift_left(int_board)
    down_left = _shift_left(down)
    down_right = _shift_right(down)
    up_left = _shift_left(up)
    up_right = _shift_right(up)
    return down + up + right + left + down_left + down_right + up_left + up_right

def update_board(board):
    """
    Apply the update rules to the board.
    """
    counts = count_neighbors(board)
    is_three = tf.equal(counts, 3)
    stay_alive = tf.logical_and(board, tf.logical_or(tf.equal(counts, 2), is_three))
    come_to_life = tf.logical_and(tf.logical_not(board), is_three)
    return tf.logical_or(stay_alive, come_to_life)

def create_icolumn(board, row, col):
    """
    Create the "icolumn" formation, which has cyclic order
    15 and looks like an I at some point.
    """
    for i in range(5, 13):
        for j in range(4, 7):
            if j != 5 or (i != 6 and i != 11):
                board[i + row, j + col] = True

def print_board(board):
    for i, row in enumerate(board):
        for value in row:
            if value:
                sys.stdout.write('#')
            else:
                sys.stdout.write(' ')
        sys.stdout.write('\n')
    sys.stdout.flush()

if __name__ == '__main__':
    main()

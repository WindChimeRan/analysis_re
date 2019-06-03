import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, List

from scipy.special import comb, perm
from mpl_toolkits.mplot3d import Axes3D


# calculate the probability of accuracy(correct_triplet, triplet_num) for seq2seq and tagging


def p_seq2seq(correct_p: float, triplet_num: int, correct_triplet: int, not_less_mode: bool) -> float:

    assert correct_triplet <= triplet_num

    if correct_triplet == 0:
        result = 1 - correct_p ** 3

    if correct_triplet < triplet_num:
        result = correct_p ** (correct_triplet * 3) * (1 - correct_p)
    else:
        result = correct_p ** (correct_triplet * 3)
    if not not_less_mode or correct_triplet == triplet_num:
        return result
    else:
        return result + p_seq2seq(correct_p, triplet_num, correct_triplet + 1, not_less_mode)

def p_tagging(correct_p: float, triplet_num: int, correct_triplet: int, not_less_mode: bool) -> float:

    assert correct_triplet <= triplet_num

    result = ((correct_p ** 3) ** correct_triplet) * comb(triplet_num, correct_triplet) * ((1 - correct_p ** 3) ** (triplet_num - correct_triplet))
    if not not_less_mode or correct_triplet == triplet_num:
        return result
    else:
        return result + p_tagging(correct_p, triplet_num, correct_triplet + 1, not_less_mode)

def plot_basic(triplet_num: int, not_less_mode: bool) -> None:
    # plot correct_triplet  0/5 -- 5/5
    # plot correct_p        0.5 -- 0.9
    # plot probability

    # for x y:
    x = list(range(1, 6))         # correct_triplet
    # y = [0.5, 0.6, 0.7, 0.8, 0.9] # correct_p
    y = list(np.arange(0.5, 1.0, 0.01)) # correct_p
    def calculate_z_for_xy(x: List[int], y: List[float], f: Callable[[float, int, int, bool], float]) -> List[float]:

        result: float = []
        for xx in x:
            for yy in y:
                result.append(f(correct_p=yy, triplet_num=triplet_num, correct_triplet=xx, not_less_mode=not_less_mode))
        return result

    
    z_seq2seq = np.array(calculate_z_for_xy(x, y, p_seq2seq))
    z_tagging = np.array(calculate_z_for_xy(x, y, p_tagging))
    x_plot = np.array([xx for xx in x for yy in y])
    y_plot = np.array([yy for xx in x for yy in y])

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.scatter(x_plot, y_plot, z_seq2seq, c='r', label='seq2seq')
    # ax.scatter(x_plot, y_plot, z_tagging, c='b', label='tagging')
    ax.scatter(x_plot, y_plot, z_tagging/z_seq2seq, c='b', label='tagging/seq2seq')

    ax.legend(loc='best')

    ax.set_zticks([1, 2, 5, 10, 14], minor=False)
    ax.set_xticks([1, 2, 3, 4, 5], minor=False)

    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('correct p', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('correct triplet num', fontdict={'size': 15, 'color': 'red'})

    plt.show()


if __name__ == "__main__":

    correct_p = 0.9
    sentence_len = 100
    triplet_num = 5
    correct_triplet = 5
    not_less_mode = True

    p_copy_re_joint_right = p_seq2seq(correct_p, triplet_num, correct_triplet, not_less_mode)
    p_tagging_right = p_tagging(correct_p, triplet_num, correct_triplet, not_less_mode)
    print(p_copy_re_joint_right)
    print(p_tagging_right)
    print(p_tagging_right/p_copy_re_joint_right)

    plot_basic(triplet_num, not_less_mode)
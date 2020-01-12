def my_agent(observation, configuration):
    import numpy as np
    from scipy.signal import convolve2d

    # color
    my_color = observation['mark']
    opponent_color = 1 if my_color == 2 else 2

    # In a row
    n = configuration['inarow']

    # Convolutions
    hori = np.ones(shape=(n, 1))
    verti = np.ones(shape=(1, n))
    ddown = np.eye(N=n)
    dup = np.eye(N=n)[::-1]

    def score_board(obs_matrix, turn):

        def in_a_row(obs_matrix, color=1):
            color_matrix = (obs_matrix == color).astype(int)
            color_matrix_potential = ((obs_matrix == color) | (obs_matrix == 0)).astype(int)
            open_1 = 0
            open_2 = 0
            open_3 = 0
            open_4 = 0

            for conv in [hori, verti, ddown, dup]:
                inarow = convolve2d(color_matrix, conv)
                inaopenrow = convolve2d(color_matrix_potential, conv)
                connection_potential = (inaopenrow == n).astype(int) * inarow

                open_1 += (connection_potential == 1).sum()
                open_2 += (connection_potential == 2).sum()
                open_3 += (connection_potential == 3).sum()
                open_4 += (connection_potential == 4).sum()

            eps = 0.01
            if (color == my_color and turn == 1) or (color == opponent_color and turn == -1):
                color_score = 2 * open_1 + (3 + eps) * open_2 + (100 + eps) * open_3 + 100 * open_4
            else:
                color_score = 1 * open_1 + (2 + eps) * open_2 + (4 + eps) * open_3 + 100 * open_4

            episode_end = True if open_4 > 0 else False

            return color_score, episode_end

        score_1, episode_end_1 = in_a_row(obs_matrix, color=my_color)
        if episode_end_1:
            return score_1, True
        else:
            score_2, episode_end_2 = in_a_row(obs_matrix, color=opponent_color)
            if episode_end_2:
                return -score_2, True
            else:
                final_score = score_1 - score_2
                return final_score, False

    def drop_coin(obs_matrix, column, my_color):
        obs_copy = obs_matrix.copy()
        i = 0
        for rowval in obs_copy[:, column]:
            if rowval == 0:
                row = i
            i += 1
        obs_copy[row, column] = my_color
        return obs_copy

    def find_best_move(obs_matrix, turn=-1, max_depth=5, depth=0):
        turn *= -1
        depth += 1
        obs_copy = obs_matrix.copy()
        best_score = -999 * turn
        for col_num in range(configuration['columns']):
            if (obs_copy[:, col_num] == 0).astype(int).sum() == 0:
                continue
            if turn == 1:
                coin_color = my_color
            else:
                coin_color = opponent_color
            obs_copy_move = drop_coin(obs_copy, col_num, coin_color)
            if depth == max_depth:
                score, episode_end = score_board(obs_copy_move, turn)
            else:
                current_score, episode_end = score_board(obs_copy_move, turn)
                if episode_end:
                    best_score = current_score
                    best_column = col_num
                    break
                else:
                    score = find_best_move(obs_copy_move, turn=turn, depth=depth)
            if turn == 1 and score > best_score:
                best_score = score
                best_column = col_num
            elif turn == -1 and score < best_score:
                best_score = score
                best_column = col_num
        if depth == 1:
            return best_column
        else:
            return best_score

    # Calculate
    obs_matrix = np.array(observation['board']).reshape(configuration['rows'], configuration['columns'])
    best_column = find_best_move(obs_matrix)

    return best_column
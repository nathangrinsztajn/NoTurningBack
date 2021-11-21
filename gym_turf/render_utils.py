import numpy as np
import pkg_resources
import imageio

def room_to_tiny_world_rgb(room, player_position, scale=1):
    room = np.array(room)

    grass = [77, 197, 17]
    ruined_grass = [160, 96, 13]
    goal = [254, 126, 125]
    stone_path = [177, 175, 171]
    agent = [0, 126, 252]

    surfaces = [grass, ruined_grass, stone_path, agent, goal]

    # Assemble the new rgb_room, with all loaded images
    room_small_rgb = np.zeros(shape=(room.shape[0] * scale, room.shape[1] * scale, 3), dtype=np.uint8)
    for i in range(room.shape[0]):
        x_i = i * scale
        for j in range(room.shape[1]):
            y_j = j * scale
            surfaces_id = int(room[i, j])
            room_small_rgb[x_i:(x_i + scale), y_j:(y_j + scale), :] = np.array(surfaces[surfaces_id])
    room_small_rgb[player_position[0], player_position[1]] = agent
    return room_small_rgb

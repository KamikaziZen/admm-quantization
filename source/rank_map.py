# Rank Configuratins Based on error delta and decomposition


def get_rank_map(eps, decomp):
    if eps == 0.005 and decomp == 'cp3':
        return {
            'layer1.0.conv1': 61,
            'layer1.0.conv2': 55,
            'layer1.1.conv1': 73,
            'layer1.1.conv2': 63,
            'layer2.0.conv1': 131,
            'layer2.0.conv2': 127,
            'layer2.1.conv1': 157,
            'layer2.1.conv2': 113,
            'layer3.0.conv1': 145,
            'layer3.0.conv2': 223,
            'layer3.1.conv1': 231,
            'layer3.1.conv2': 220,
            'layer4.0.conv1': 307,
            'layer4.0.conv2': 644,
            'layer4.1.conv1': 505,
            'layer4.1.conv2': 248 
        }
    elif eps == 0.003 and decomp == 'cp3':
        return {
            'layer1.0.conv1': 64,
            'layer1.0.conv2': 61,
            'layer1.1.conv1': 78,
            'layer1.1.conv2': 73,
            'layer2.0.conv1': 133,
            'layer2.0.conv2': 146,
            'layer2.1.conv1': 173,
            'layer2.1.conv2': 131,
            'layer3.0.conv1': 159,
            'layer3.0.conv2': 248,
            'layer3.1.conv1': 268,
            'layer3.1.conv2': 249,
            'layer4.0.conv1': 363,
            'layer4.0.conv2': 752,
            'layer4.1.conv1': 574,
            'layer4.1.conv2': 282
        }
    elif eps == 0.001 and decomp == 'cp3':
        return {
            'layer1.0.conv1': 74,
            'layer1.0.conv2': 67,
            'layer1.1.conv1': 93,
            'layer1.1.conv2': 81,
            'layer2.0.conv1': 169,
            'layer2.0.conv2': 211,
            'layer2.1.conv1': 225,
            'layer2.1.conv2': 169,
            'layer3.0.conv1': 178,
            'layer3.0.conv2': 317,
            'layer3.1.conv1': 329,
            'layer3.1.conv2': 363,
            'layer4.0.conv1': 399,
            'layer4.0.conv2': 964,
            'layer4.1.conv1': 787,
            'layer4.1.conv2': 341
        }
    elif eps == 0.0005 and decomp == 'cp3':
        return {
            'layer1.0.conv1': 67,
            'layer1.0.conv2': 74,
            'layer1.1.conv1': 97,
            'layer1.1.conv2': 83,
            'layer2.0.conv1': 165,
            'layer2.0.conv2': 212,
            'layer2.1.conv1': 243,
            'layer2.1.conv2': 181,
            'layer3.0.conv1': 189,
            'layer3.0.conv2': 379,
            'layer3.1.conv1': 367,
            'layer3.1.conv2': 364,
            'layer4.0.conv1': 451,
            'layer4.0.conv2': 1092,
            'layer4.1.conv1': 857,
            'layer4.1.conv2': 347
        }
    elif eps == 0.0001 and decomp == 'cp3':
        return {
            'layer1.0.conv1': 80,
            'layer1.0.conv2': 81,
            'layer1.1.conv1': 99,
            'layer1.1.conv2': 86,
            'layer2.0.conv1': 176,
            'layer2.0.conv2': 264,
            'layer2.1.conv1': 263,
            'layer2.1.conv2': 202,
            'layer3.0.conv1': 209,
            'layer3.0.conv2': 453,
            'layer3.1.conv1': 439,
            'layer3.1.conv2': 468,
            'layer4.0.conv1': 540,
            'layer4.0.conv2': 1147,
            'layer4.1.conv1': 969,
            'layer4.1.conv2': 581
        }
    elif eps == 0.0005 and decomp == 'cp3-epc':
        return {
            'layer1.0.conv1': 101,
            'layer1.0.conv2': 90,
            'layer1.1.conv1': 133,
            'layer1.1.conv2': 107,
            'layer2.0.conv1': 249,
            'layer2.0.conv2': 293,
            'layer2.1.conv1': 302,
            'layer2.1.conv2': 212,
            'layer3.0.conv1': 247,
            'layer3.0.conv2': 438,
            'layer3.1.conv1': 492,
            'layer3.1.conv2': 396,
            'layer4.0.conv1': 605,
            'layer4.0.conv2': 1300,
            'layer4.1.conv1': 1200,
            'layer4.1.conv2': 850,
        }
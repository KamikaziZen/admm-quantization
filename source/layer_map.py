def get_layer_list(model_name, **kwargs):
    
    layer_list = []
    
    if model_name == 'unet':
        layer_list.extend(f'encoder{i}.enc{i}conv{j}' for i in range(1, 5) for j in range(1, 3))
        layer_list.extend(f'bottleneck.bottleneckconv{j}' for j in range(1, 3))
        layer_list.extend(f'decoder{i}.dec{i}conv{j}' for i in range(1, 5) for j in range(1, 3))
    
    if model_name == 'resnet18':
        for module in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer_list.extend([f'{module}.0.conv1', f'{module}.0.conv2', 
                               f'{module}.1.conv1', f'{module}.1.conv2'])
        if kwargs.get('downsample'):
            for module in ['layer2', 'layer3', 'layer4']:
                layer_list.append(f'{module}.0.downsample.0')
                
        if kwargs.get('conv1'):
            layer_list.append('conv1')
            
        if kwargs.get('fc'):
            layer_list.append('fc')
            
    elif model_name == 'resnet50':
        jmap = {
            1: 3,
            2: 4,
            3: 6, 
            4: 3
        }
        layer_list = [f'layer{i}.{j}.conv{k}' for i in range(1, 5) for j in range(jmap[i]) for k in range(1,4)]
        
    elif model_name == 'deit':
        for i in range(12):
            layer_list.extend([f'blocks.{i}.attn.qkv',
                               f'blocks.{i}.attn.proj',
                               f'blocks.{i}.mlp.fc1',
                               f'blocks.{i}.mlp.fc2'])
                
    elif model_name == 'regnet_y_3_2gf':
        jmap = {
            1: 1,
            2: 4,
            3: 12,
            4: 0
        }

        for i in [1,2,3,4]:
            layer_list.extend(
                    (f'trunk_output.block{i}.block{i}-0.proj.0',
                     f'trunk_output.block{i}.block{i}-0.f.a.0',
#                      f'trunk_output.block{i}.block{i}-0.f.b.0',
                     f'trunk_output.block{i}.block{i}-0.f.se.fc1',
                     f'trunk_output.block{i}.block{i}-0.f.se.fc2',
                     f'trunk_output.block{i}.block{i}-0.f.c.0')
            )
            for j in range(1, jmap[i]+1):
                layer_list.extend(
                     (f'trunk_output.block{i}.block{i}-{j}.f.a.0',
#                       f'trunk_output.block{i}.block{i}-{j}.f.b.0',
                      f'trunk_output.block{i}.block{i}-{j}.f.se.fc1',
                      f'trunk_output.block{i}.block{i}-{j}.f.se.fc2',
                      f'trunk_output.block{i}.block{i}-{j}.f.c.0')
                )
                
    elif model_name == 'regnet_y_400mf':
        jmap = {
            1: 0,
            2: 2,
            3: 5,
            4: 5
        }

        for i in [1,2,3,4]:
            layer_list.extend(
                    (f'trunk_output.block{i}.block{i}-0.proj.0',
                     f'trunk_output.block{i}.block{i}-0.f.a.0',
                     f'trunk_output.block{i}.block{i}-0.f.b.0',
                     f'trunk_output.block{i}.block{i}-0.f.se.fc1',
                     f'trunk_output.block{i}.block{i}-0.f.se.fc2',
                     f'trunk_output.block{i}.block{i}-0.f.c.0')
            )
            for j in range(1, jmap[i]+1):
                layer_list.extend(
                     (f'trunk_output.block{i}.block{i}-{j}.f.a.0',
                      f'trunk_output.block{i}.block{i}-{j}.f.b.0',
                      f'trunk_output.block{i}.block{i}-{j}.f.se.fc1',
                      f'trunk_output.block{i}.block{i}-{j}.f.se.fc2',
                      f'trunk_output.block{i}.block{i}-{j}.f.c.0')
                )
                
        if kwargs.get('stem'):
            layer_list.append('stem.0')
                
        if kwargs.get('fc'):
            layer_list.append('fc')
                
    return layer_list
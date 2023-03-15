def get_layer_list(model_name, **kwargs):
    
    layer_list = []
    
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
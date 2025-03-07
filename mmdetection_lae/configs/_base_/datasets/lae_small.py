# dataset settings
dataset_type = 'CocoDataset'

backend_args = None
# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]

#================SLM==================
data_root = '../data/LAE-COD/SLM/'
metainfo = dict(
    classes =  ('Airplane', 'Airplanes', 'Airport', 'Airport runway', 'Amphitheater', 'Baseball field', 'Bridge', 'Building', 'Building complex', 'Building with parking lot', 'Burnt area', 'Burnt vegetation', 'Coastal erosion', 'Coastal infrastructure', 'Commercial building', 'Construction equipment', 'Construction site', 'Container crane', 'Container ship', 'Container storage area', 'Container yard', 'Crosswalk', 'Dam', 'Farm', 'Farm fields', 'Farm land', 'Farm machinery', 'Farm or Agricultural Land', 'Farm or agricultural facility', 'Field', 'Football field', 'Forest', 'Forested area', 'Golf course', 'Grass', 'Grass field', 'Grassland', 'Harbor', 'Highway', 'Industrial area', 'Industrial building', 'Industrial buildings', 'Industrial complex', 'Industrial facility', 'Industrial or agricultural complex', 'Industrial or commercial building', 'Industrial or commercial complex', 'Industrial port', 'Industrial warehouse', 'Lake', 'Lake or pond', 'Marina', 'Marine structure', 'Military base', 'Military fortification', 'Military vessel', 'Mobile homes', 'Natural landscape', 'Natural water body', 'Oil tanker', 'Open field', 'Park', 'Park or public square', 'Park or recreational area', 'Parking lot', 'Paved area', 'Railroad tracks', 'Residential area', 'Residential buildings', 'Road', 'Road construction', 'Road intersection', 'Roadway', 'Roof', 'Row crops', 'Runway', 'Rural area', 'Rural landscape', 'Rural residential area', 'Rural settlement', 'Salt evaporation ponds', 'School', 'Ship', 'Shipyard', 'Soccer field', 'Solar panel array', 'Solar panel installation', 'Sports court', 'Sports facilities', 'Sports field', 'Stadium', 'Storage containers', 'Storage tank', 'Storage tanks', 'Swimming pool', 'Tennis court', 'Tennis courts', 'Track and field', 'Track and field stadium', 'Urban area', 'Vegetation', 'Vehicles', 'Water bodies', 'Water body', 'Water storage tank', 'Water treatment facility')
)
SLM_train_dataset=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='SLM.json',
    metainfo=metainfo,
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=backend_args)

#================Power-Plant==================
data_root = '../data/LAE-FOD/Power-Plant/'
metainfo = dict(
    classes = ('working condensing tower', 'unworking condensing tower', 'working chimney', 'unworking chimney')
)
PowerPlant_train_dataset=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='processed_LAE-1M_Power-Plant_train.json',
    metainfo=metainfo,
    data_prefix=dict(img='Images/'),
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=backend_args)

#================LAE-80C==================
data_root = '../data/LAE-80C/'
metainfo = dict(classes = ('airplane', 'airport', 'groundtrackfield', 'harbor', 'baseballfield', 'overpass', 'basketballcourt', 'bridge', 'stadium', 'storagetank', 'tenniscourt', 'expressway service area', 'trainstation', 'expressway toll station', 'vehicle', 'golffield', 'windmill', 'dam', 'helicopter', 'roundabout', 'soccer ball field', 'swimming pool', 'container crane', 'helipad', 'Bus', 'Cargo Truck', 'Dry Cargo Ship', 'Dump Truck', 'Engineering Ship', 'Excavator', 'Fishing Boat', 'Intersection', 'Liquid Cargo Ship', 'Motorboat', 'Passenger Ship', 'Small Car', 'Tractor', 'Trailer', 'Truck Tractor', 'Tugboat', 'Van', 'Warship', 'working condensing tower', 'unworking condensing tower', 'working chimney', 'unworking chimney', 'Fixed-wing Aircraft', 'Small Aircraft', 'Cargo Plane', 'Pickup Truck', 'Utility Truck', 'Passenger Car', 'Cargo Car', 'Flat Car', 'Locomotive', 'Sailboat', 'Barge', 'Ferry', 'Yacht', 'Oil Tanker', 'Engineering Vehicle', 'Tower crane', 'Reach Stacker', 'Straddle Carrier', 'Mobile Crane', 'Haul Truck', 'Front loader/Bulldozer', 'Cement Mixer', 'Ground Grader', 'Hut/Tent', 'Shed', 'Building', 'Aircraft Hangar', 'Damaged Building', 'Facility', 'Construction Site', 'Shipping container lot', 'Shipping Container', 'Pylon', 'Tower'))
LAE80C_val_dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='LAE-80C-benchmark.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        metainfo=metainfo,
        pipeline=test_pipeline,
        return_classes=True,
        backend_args=backend_args)
LAE80C_val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'LAE-80C-benchmark.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

#############
dataset_prefixes = ['DIOR', 'DOTAv2', 'LAE-80C']

##### ALL LAE-COD + LAE-FOD
all_train_dataset = [ SLM_train_dataset, PowerPlant_train_dataset]

all_val_dataset = [LAE80C_val_dataset]
all_metrics = [LAE80C_val_evaluator]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='ConcatDataset', datasets=all_train_dataset))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='ConcatDataset', datasets=all_val_dataset))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=all_metrics,
    dataset_prefixes=dataset_prefixes)
test_evaluator = val_evaluator
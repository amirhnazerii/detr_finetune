# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .detr_robust import build as build_robust

def build_model(args):
    return build_robust(args) if args.robust else build(args) 

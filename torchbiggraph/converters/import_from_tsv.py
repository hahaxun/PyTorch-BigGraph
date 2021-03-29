#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import argparse
from mpi4py import MPI
from pathlib import Path

from torchbiggraph.config import ConfigFileLoader, ConfigSchema
from torchbiggraph.converters.importers import (
    TSVEdgelistReader,
    PytablesEdgelistReader,
    convert_input_data,
    parse_config_partial,
)


def main():
    config_help = "\n\nConfig parameters:\n\n" + "\n".join(ConfigSchema.help())
    parser = argparse.ArgumentParser(
        epilog=config_help,
        # Needed to preserve line wraps in epilog.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("-p", "--param", action="append", nargs="*")
    parser.add_argument("edge_paths", type=Path, nargs="*", help="Input file paths")
    parser.add_argument(
        "-l",
        "--lhs-col",
        type=int,
        required=True,
        help="Column index for source entity",
    )
    parser.add_argument(
        "-r",
        "--rhs-col",
        type=int,
        required=True,
        help="Column index for target entity",
    )
    parser.add_argument("--rel-col", type=int, help="Column index for relation entity")
    parser.add_argument(
        "--relation-type-min-count",
        type=int,
        default=1,
        help="Min count for relation types",
    )
    parser.add_argument(
        "--entity-min-count", type=int, default=1, help="Min count for entities"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for dpp"
    )

    parser.add_argument(
        "--base_rank", type=int, default=0, help="local rank for dpp"
    )
    opt = parser.parse_args()

    rank = MPI.COMM_WORLD.rank 
    loader = ConfigFileLoader()

    config_dict = loader.load_raw_config(opt.config, opt.param)

    (
        entity_configs,
        relation_configs,
        entity_path,
        edge_paths,
        dynamic_relations,
    ) = parse_config_partial(  # noqa
        config_dict
    )

    
    convert_input_data(
        entity_configs,
        relation_configs,
        entity_path,
        edge_paths,
        opt.edge_paths,
        PytablesEdgelistReader(opt.lhs_col, opt.rhs_col, opt.rel_col),
        opt.entity_min_count,
        opt.relation_type_min_count,
        dynamic_relations,
        True,
        opt.base_rank + rank#opt.local_rank
    )


if __name__ == "__main__":
    main()

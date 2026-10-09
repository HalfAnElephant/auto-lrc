#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path
import re


def to_standard_lrc(enhanced_lrc):
    """将增强型LRC（含逐字<mm:ss.xxx>时间戳）转为标准LRC（仅保留行级时间戳）"""
    return re.sub(r'<\d+:\d{2}\.\d{3}>', '', enhanced_lrc)


def build_parser():
    parser = argparse.ArgumentParser(description='自动打轴工具')
    parser.add_argument('lrc_file', help='歌词文本')
    parser.add_argument('music_file', help='歌曲文件')
    parser.add_argument('-f', '--format', choices=('lrc',), default='lrc',
                        help='输出文件格式，仅支持lrc（默认）')
    parser.add_argument('-l', '--line_only', type=int, choices=(0, 1), default=0,
                        help='1:逐句/0:逐字, 默认：逐字')
    parser.add_argument('-v', '--vocalize', type=int, choices=(0, 1), default=1,
                        help='1:分离人声/0:不分离, 默认：分离')
    parser.add_argument(
        '-m', '--model', choices=('mdx', 'mdx_extra', 'mdx_q', 'mdx_extra_q'),
        help='demucs模型，默认：mdx_extra', default='mdx_extra')
    parser.add_argument(
        '-i', '--idx', type=int, choices=(-1, 0, 1, 2, 3),
        help='demucs模型idx, 默认：-1', default=-1)
    parser.add_argument('-o', '--out_dir', type=str, default='demofile',
                        help='标准LRC输出目录，默认：demofile')
    parser.add_argument('--aligner', choices=('mtl', 'mtl_bdr'), default='mtl',
                        help='粗对齐模型，默认：mtl；mtl_bdr增加行边界检测')
    parser.add_argument('--json-output', metavar='PATH',
                        help='保存原始帧、浮点秒和未校准路径分数的JSON文件')
    parser.add_argument('--enhanced-lrc-output', metavar='PATH',
                        help='另外保存当前逐字/逐句LRC输出')
    return parser


def _check_output_paths(parser, input_paths, output_paths):
    seen = [Path(path).resolve() for path in input_paths]
    for output_path in output_paths:
        path = Path(output_path).resolve()
        for other in seen:
            if path == other or (path.exists() and other.exists() and path.samefile(other)):
                parser.error('输出路径与输入或其他输出冲突: {}'.format(output_path))
        seen.append(path)


def cli():
    parser = build_parser()
    args = parser.parse_args()
    base_name = os.path.splitext(os.path.basename(args.music_file))[0]
    out_path = os.path.join(args.out_dir, base_name + '.lrc')
    output_paths = [out_path]
    output_paths.extend(path for path in (args.json_output, args.enhanced_lrc_output) if path)
    _check_output_paths(parser, [args.lrc_file, args.music_file], output_paths)

    import ext.t2lutils as t2lutils
    from t2l.t2l import process

    txt_lines = None
    encoding = t2lutils.__get_file_encoding(args.lrc_file)
    with open(args.lrc_file, 'r', encoding=encoding) as fp:
        txt_lines = fp.readlines()
    result = process(
        txt_lines,
        args.music_file,
        mtl_model='MTL_BDR' if args.aligner == 'mtl_bdr' else 'MTL',
        demucs_model=args.model,
        demucs_idx=args.idx,
        format=args.format,
        line_only=(args.line_only == 1),
        vocalize=(args.vocalize == 1),
        return_details=bool(args.json_output),
    )
    lrc = result['lrc'] if args.json_output else result
    print(lrc)

    # 转为标准LRC并保存到输出目录
    standard_lrc = to_standard_lrc(lrc)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(standard_lrc)
    print(f'\n标准LRC已保存至: {out_path}')

    if args.enhanced_lrc_output:
        Path(args.enhanced_lrc_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.enhanced_lrc_output).write_text(lrc, encoding='utf-8')
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')


if __name__ == '__main__':
    cli()

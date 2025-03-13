import sys, os
import argparse

dir_name = os.path.dirname(__file__)
sys.path.append(dir_name)

from functions import format_residue_single,format_residue_multi,format_4lines,TimingsManager
output_format_dic = {"residue_single":format_residue_single,"residue_multi":format_residue_multi,"4lines":format_4lines}

from models import DARUMA


def main():

    parser = argparse.ArgumentParser(description='Process input and output files.')
    parser.add_argument('input_file', help='Path to input file')
    parser.add_argument('-o', '--output', dest='output_file', default='daruma.out', help='Path to output file')
    parser.add_argument('--output-format', dest='output_format', default='4lines', help='Specify the format of the output results')
    parser.add_argument("--no-smoothing", dest='smoothing', action="store_false", help="Turn off smoothing")
    parser.add_argument("--no-remove-short-regions", dest='remove_short_regions', action="store_false", help="Turn off short area merge")
    parser.add_argument("--gpu-id", dest='gpu_id', default="0", help="ID of GPU to be used")
    args = parser.parse_args()

    input_path = args.input_file
    output_path = args.output_file
    output_format = args.output_format
    smoothing = args.smoothing
    remove_short_regions = args.remove_short_regions
    gpu_id = args.gpu_id

    # DARUMA呼び出し
    daruma_model = DARUMA(gpu_id)
    
    # 入力
    with open(input_path,"r") as f:
        data = f.read().strip(">").split("\n>")

    # 出力
    timings_manager = TimingsManager()
    result_file_manager = output_format_dic[output_format](output_path)

    # 閾値
    threshold = 0.5

    if smoothing:
        smoothing = 17

    replace_map = str.maketrans("UBJOZ", "XXXXX", "\n ")

    for block in data:
        timings_manager.start()
        ac,seq = block.split("\n",1)        
        seq = seq.translate(replace_map)

        pred_prob,pred_class = daruma_model.predict_from_sequence(seq,threshold=threshold,smoothing_window=smoothing,remove_short_regions=remove_short_regions)

        result_file_manager.append_write(ac,seq,pred_prob,pred_class)
        timings_manager.end(ac)

    result_file_manager.close_manager()

    return


if __name__ == "__main__":
    main()


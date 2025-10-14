import argparse

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="The score collection process.");

    # Argument for the input file path
    parser.add_argument(
        "--file_path","-fp",type=str,default="",
        help="The path to the input file."
    );

    # Argument for the input file name
    parser.add_argument(
        "--input_file","-if",type=str,default="",
        help="The name of the input file."
    );

    # Argument for the graph file
    parser.add_argument(
        "--graph_file","-gf",type=str,default="",
        help="The name of the graph file."
    );
    
    # The number of processes to start for the task.
    parser.add_argument(
        "--number_of_processes",
        "-nop",
        type=int,
        help="The number of processes to start for the task."
    )
    
    # Parse the arguments
    args = parser.parse_args();

    file_path = args.file_path;
    input_file = args.input_file;
    graph_edge_file = args.graph_file;

    if args.number_of_processes:
        print("--- number_of_processes ---")
        print(f"number_of_processes: {args.number_of_processes}")
    
    full_info = f"{file_path} - {input_file} - {graph_edge_file}";
    print(full_info);

if __name__ == "__main__":
    #python TestArguments.py --file_path SimulationData/Mixed/Test/ --input_file mixed_sim_data.csv --graph_file mixed_sim_graph_edges.csv --number_of_processes=6
    
    main()
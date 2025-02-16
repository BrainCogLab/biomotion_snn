import os
def save_figure_as_pdf(figure, save_dir, filename):
    # Remove the file extension
    file_name_without_extension = os.path.splitext(filename)[0]

    # Get the current path and parent path
    current_path = os.getcwd()
    parent_path = os.path.dirname(current_path)

    # Define the save path
    save_path = os.path.join(parent_path, save_dir)

    # If the save path doesn't exist, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print("保存文件路径已存在")

    # Save the figure
    save_file_path = os.path.join(save_path, filename)
    figure.savefig(save_file_path, dpi=300, bbox_inches='tight', format='pdf')

    print("Diagram saved successfully at:", save_file_path)
import os
import json

def extract_keypoints(directory):
    keypoints_list = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            
            try:
                with open(filepath, 'r') as file:
                    data = json.load(file)
                    people = data.get("people", [])
                    
                    if people:
                        keypoints = people[0].get("pose_keypoints_2d", [])
                        if keypoints:
                            first_two = keypoints[:2]
                            keypoints_list.append(first_two)
            
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    print(f"Processed {len(keypoints_list)} files")

    return keypoints_list

def get_nodding(set_num=0):
    # Replace 'your_directory_path' with the path to your directory
    directory_path = f'C:\\Users\\xueyu\\Desktop\\evasion\\dataset{str(set_num)}\\nodding_json'
    result = extract_keypoints(directory_path)
    return result

def get_shaking(set_num=0):
    # Replace 'your_directory_path' with the path to your directory
    directory_path = f'C:\\Users\\xueyu\\Desktop\\evasion\\dataset{str(set_num)}\\shaking_json'
    result = extract_keypoints(directory_path)
    return result

def get_s_n(set_num=1):
    # Replace 'your_directory_path' with the path to your directory
    directory_path = f'C:\\Users\\xueyu\\Desktop\\evasion\\dataset{str(set_num)}\\s_n_json'
    result = extract_keypoints(directory_path)
    return result

def write_into_txt(target, name):
    with open(f"C:\\Users\\xueyu\\Desktop\\evasion\\dataset\\{name}.txt", "w") as file:
        for pairs in target:
            for pair in pairs:
                file.write(f"{pair}\n")

if __name__ == "__main__":
    nodding = get_nodding()
    shaking = get_shaking()
    write_into_txt(nodding, "nodding")
    write_into_txt(shaking, "shaking")
import json

def step1(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            # Parse the JSON line
            data = json.loads(line)

            # Extract required data, with a check for missing keys
            extracted_data = {
                "train_ep_max/success": data.get("train_ep_max/success"),
                "_step": data.get("_step"),
                "test_ep/success": data.get("test_ep/success")
            }

            # Write the extracted data to the output file
            output_file.write(json.dumps(extracted_data) + '\n')

    print(f"Data extracted to {output_file_path}")

def step2(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        line_count = 0
        success_count = 0
        sections = []

        for line in input_file:
            data = json.loads(line)


            # Count line and successes
            line_count += 1
            if data.get("train_ep_max/success") == 1:
                success_count += 1


            if line_count % 10 == 0:
                success_rate = success_count / line_count if line_count > 0 else 0
                sections.append((data["_step"], success_rate))

                # Reset counters for the next section
                #line_count = 0
                #success_count = 0


        # Write the sections data to output file
        for section in sections:
            output_file.write(json.dumps({"_step": section[0], "success_rate": section[1]}) + '\n')

    print(f"Success rates calculated and saved to {output_file_path}")

if __name__ == "__main__":

    input_raw_files = [""]
    step1_output_files = [""]
    step2_output_files = [""]

    assert(len(input_raw_files) == len(step1_output_files)) 
    assert(len(step1_output_files) == len(step2_output_files))
    
    file_num = len(input_raw_files)
    for i in range(file_num):
        input_file_path = input_raw_files[i]
        step1_output = step1_output_files[i]
        step2_output = step2_output_files[i]


        step1(input_file_path=input_file_path, output_file_path=step1_output)
        step2(input_file_path=step1_output, output_file_path=step2_output)
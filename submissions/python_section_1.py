from typing import Dict, List, Any
import pandas as pd
import re
import polyline
from math import radians, sin, cos, sqrt, atan2

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    result = [] 
    for i in range(0, len(lst), n):
        group = lst[i:i+n] 
        reversed_group = []
        for j in range(len(group)-1, -1, -1):
            reversed_group.append(group[j])
        # adding the reversed group to the result
        result.extend(reversed_group) 
    return result
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))



def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    length_dict = {}
    for string in lst:
        length = len(string) 
        if length not in length_dict:
            length_dict[length] = []  
        length_dict[length].append(string)

    return dict(sorted(length_dict.items()))
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))  
 

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    flattened = {} 

    def flatten_helper(current_dict: Dict[str, Any], parent_key: str):
        
        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key 
            if isinstance(value, dict): 
                flatten_helper(value, new_key)
            elif isinstance(value, list):  
                for index, item in enumerate(value):
                    item_key = f"{new_key}[{index}]"  
                    if isinstance(item, dict):  
                        flatten_helper(item, item_key)
                    else:  
                        flattened[item_key] = item
            else: 
                flattened[new_key] = value

    flatten_helper(nested_dict, '')  
    return flattened

test_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}
flattened_dict = flatten_dict(test_dict)
print(flattened_dict)

def unique_permutations(nums: List[int]) -> List[List[int]]:
    nums.sort() 
    result = []  
    used = [False] * len(nums)  
    
    def backtrack(current_permutation: List[int]):
        #if the current permutation is of the same length as nums
        if len(current_permutation) == len(nums):
            result.append(current_permutation[:])  
            return
        
        for i in range(len(nums)):
            if used[i]:
                continue 
            
            #only allowing the first occurrence of each number in the current position
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue
            
            used[i] = True
            current_permutation.append(nums[i])
            backtrack(current_permutation)  
            current_permutation.pop()  
            used[i] = False            
    backtrack([])  #backtracking with an empty current permutation
    return result
input_nums = [1, 1, 2]
unique_perms = unique_permutations(input_nums)
print(unique_perms)


def find_all_dates(text: str) -> List[str]:
    date_pattern = r'(?:(\d{2})-(\d{2})-(\d{4})|(\d{2})/(\d{2})/(\d{4})|(\d{4})\.(\d{2})\.(\d{2}))'
    matches = re.findall(date_pattern, text)
    

    valid_dates = []
    for match in matches:
        if match[0]:  # dd-mm-yyyy
            valid_dates.append(f"{match[0]}-{match[1]}-{match[2]}")
        elif match[3]:  # mm/dd/yyyy
            valid_dates.append(f"{match[3]}/{match[4]}/{match[5]}")
        elif match[6]:  # yyyy.mm.dd
            valid_dates.append(f"{match[6]}.{match[7]}.{match[8]}")
    
    return valid_dates
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
found_dates = find_all_dates(text)
print(found_dates)
    

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    coordinates = polyline.decode(polyline_str)

    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    df['distance'] = 0.0

    R = 6371000  
    for i in range(1, len(df)):
        lat1, lon1 = radians(df.loc[i-1, 'latitude']), radians(df.loc[i-1, 'longitude'])
        lat2, lon2 = radians(df.loc[i, 'latitude']), radians(df.loc[i, 'longitude'])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        #Haversine formula
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        df.loc[i, 'distance'] = R * c
    
    return df

polyline_str = '_p~iF~ps|U_ulLnnqC_mqNvxq`@'
df = polyline_to_dataframe(polyline_str)
print(df)


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
  
    rotated_matrix = [[0] * n for _ in range(n)]  
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    transformed_matrix = [[0] * n for _ in range(n)]  
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])  
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) 
  
            transformed_matrix[i][j] = row_sum + col_sum - 2 * rotated_matrix[i][j]

    return transformed_matrix

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_multiply_matrix(matrix)
print(result)

def time_check(df) -> pd.Series:

    df['startTime'] = pd.to_datetime(df['startTime'], format='%H:%M:%S', errors='coerce').dt.time
    df['endTime'] = pd.to_datetime(df['endTime'], format='%H:%M:%S', errors='coerce').dt.time
    
    grouped = df.groupby(['id', 'id_2'])

    index = pd.MultiIndex.from_tuples(grouped.groups.keys(), names=['id', 'id_2'])
    results = pd.Series(index=index, dtype=bool)

    full_week = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}

    for name, group in grouped:
        min_time = group['startTime'].min()
        max_time = group['endTime'].max()
        full_time_coverage = (min_time <= pd.to_datetime('00:00:00').time() and 
                              max_time >= pd.to_datetime('23:59:59').time())

        days_covered = set(group['startDay']).union(set(group['endDay']))
        full_day_coverage = days_covered == full_week

        results.loc[name] = not (full_time_coverage and full_day_coverage)

    return results

df = pd.read_csv('../datasets/dataset-1.csv')
output = time_check(df)
print(output)
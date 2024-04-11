def search(arr, low, high, x):
    # Check base case
    if high >= low:

        mid = low + (high - low) // 2

        # If element is present at the middle itself
        if arr[mid] == x:
            return arr[mid]

        # If element is smaller than mid, then it can only
        # be present in left subarray
        elif arr[mid] > x:
            return search(arr, low, mid - 1, x)

        # Else the element can only be present in right subarray
        else:
            return search(arr, mid + 1, high, x)

    else:
        # Element is not present in the array
        return -1
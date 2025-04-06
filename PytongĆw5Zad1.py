def graph_maker():
    # Get input values
    len_a = int(input("Podaj wysokość pierwszego słupka (a): "))
    len_b = int(input("Podaj wysokość drugiego słupka (b): "))
    
    # Validate input conditions
    if len_a < 0 or len_a >= 9 or len_b < 0 or len_b >= 9:
        print("Wysokość słupków musi być większa lub równa 0 i mniejsza niż 9!")
        return
    
    if len_a == len_b:
        print("Wysokości słupków muszą być różne!")
        return
    
    # Calculate the height of the third column (c)
    len_c = abs(len_a - len_b)
    
    # Print the values
    print(f"a = {len_a}, b = {len_b}, c = {len_c}")
    
    # Print the bar chart
    for i in range(max(len_a, len_b, len_c) + 1):  # Start from 0 to max height
        # First column (3 columns wide)
        if i == 0:
            print("+", end="")
            print("-", end="")
            print("+", end="")
        elif i < len_a:
            print("|", end="")
        else:
            print(" ", end="")
        
        print(" ", end="")  # Single space for the gap
        
        # Second column (3 columns wide)
        if i == 0:
            print("+", end="")
            print("-", end="")
            print("+", end="")
        elif i < len_b:
            print("|", end="")
        else:
            print(" ", end="")
        
        print(" ", end="")  # Single space for the gap
        
        # Third column (3 columns wide)
        if i == 0:
            print("+", end="")
            print("-", end="")
            print("+", end="")
        elif i < len_c:
            print("|", end="")
        else:
            print(" ", end="")
        
        print()
    
# Run the program
if __name__ == "__main__":
    graph_maker()
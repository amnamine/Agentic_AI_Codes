import sys

def generate_fibonacci(n):
    """
    Generate the Fibonacci sequence up to n terms.

    Args:
        n (int): The number of terms in the sequence.

    Returns:
        list: The Fibonacci sequence as a list of integers.
    """
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        sequence = [0, 1]
        while len(sequence) < n:
            sequence.append(sequence[-1] + sequence[-2])
        return sequence

def main():
    if len(sys.argv) != 2:
        print("Usage: python fibonacci.py <number_of_terms>")
        sys.exit(1)

    try:
        n = int(sys.argv[1])
        if n < 0:
            print("Error: Number of terms must be a non-negative integer.")
            sys.exit(1)
    except ValueError:
        print("Error: Invalid input. Please enter a non-negative integer.")
        sys.exit(1)

    sequence = generate_fibonacci(n)
    print("Fibonacci sequence up to {} terms:".format(n))
    print(sequence)

if __name__ == "__main__":
    main()
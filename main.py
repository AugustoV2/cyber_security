from flask import Flask, request, jsonify
from flask_cors import CORS
import math

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Function to compute Euler's totient function
def euler_totient(n):
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result

# Function to compute modular exponentiation (a^b mod m)
def mod_exp(base, exp, mod):
    result = 1
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp //= 2
    return result

# Function to get prime factors of a number
def prime_factors(n):
    factors = set()
    p = 2
    while p * p <= n:
        while n % p == 0:
            factors.add(p)
            n //= p
        p += 1
    if n > 1:
        factors.add(n)
    return factors

# Function to check if g is a primitive root modulo n
def is_primitive_root(g, n):
    if math.gcd(g, n) != 1:
        return False  # g must be coprime to n

    phi_n = euler_totient(n)  # Compute φ(n)
    factors = prime_factors(phi_n)  # Get prime factors of φ(n)

    # Check if g^(phi_n / p) ≠ 1 (mod n) for all prime factors p
    for p in factors:
        if mod_exp(g, phi_n // p, n) == 1:
            return False
    return True

# Function to check if n has a primitive root
def has_primitive_root(n):
    if n < 2:
        return False
    if n in (2, 4):
        return True
    factors = prime_factors(n)
    if len(factors) == 1 and n % 2 != 0:
        return True  # Prime power
    if len(factors) == 2 and 2 in factors and (n // 2) in factors:
        return True  # n = 2 * p^k
    return False

# Function to find the smallest primitive root of n
def find_primitive_root(n):
    if not has_primitive_root(n):
        return -1  # No primitive root exists

    phi_n = euler_totient(n)
    factors = prime_factors(phi_n)

    for g in range(2, n):
        if math.gcd(g, n) == 1:  # Ensure g is coprime to n
            if all(mod_exp(g, phi_n // p, n) != 1 for p in factors):
                return g
    return -1  # Should never reach here if n has a primitive root

# Function to solve g^x ≡ h (mod p) using Baby-Step Giant-Step Algorithm
def baby_step_giant_step(g, h, p):
    n = math.isqrt(p) + 1  # Step size (sqrt of p)
    baby_steps = {pow(g, j, p): j for j in range(n)}
    
    # Compute g^-n using modular inverse
    g_inv_n = pow(g, p - 2, p)  # Using Fermat’s theorem for prime p
    value = h

    for i in range(n):
        if value in baby_steps:
            return i * n + baby_steps[value]  # x = i * n + j
        value = (value * g_inv_n) % p  # h * g^(-ni)
    
    return -1  # If no solution found

@app.route('/primitive-roots', methods=['POST'])
def get_primitive_roots():
    data = request.json
    n = data['prime']
    primitive_root = find_primitive_root(n)
    if primitive_root == -1:
        return jsonify({"error": f"No primitive root exists for {n}"}), 400
    return jsonify({"primitive_root": primitive_root})

@app.route('/discrete-log', methods=['POST'])
def get_discrete_log():
    data = request.json
    g = data['base']
    h = data['target']
    p = data['prime']
    x = baby_step_giant_step(g, h, p)
    if x == -1:
        return jsonify({"error": f"No solution exists for {g}^x ≡ {h} (mod {p})"}), 400
    return jsonify({"discrete_log": x})

if __name__ == '__main__':
    app.run(debug=True)

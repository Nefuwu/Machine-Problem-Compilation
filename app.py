import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import math
from fractions import Fraction as fractions
from streamlit_option_menu import option_menu
from sympy import *
import re
from array import *

st.set_page_config(
        page_title="Machine Problem Compilation",
)

#   Function for the basic math operations
def calculate_result(num1, operator, num2):
    result = None

    if operator == '+':
        result = num1 + num2
    elif operator == '-':
        result = num1 - num2
    elif operator == '*':
        result = num1 * num2
    elif operator == '/':
        result = num1 / num2
    elif operator == '**':
        result = num1 ** num2

    return result

#   Function for the absolute error
def calculate_absolute_error(actual_value, approximated_value):
    absolute_error = abs(actual_value - approximated_value)
    return absolute_error

#   Function for the relative  error
def calculate_relative_error(actual_value, approximated_value):
    relative_error = abs((actual_value - approximated_value) / actual_value)
    return relative_error

#   Converts the user input(pi,euler,sqrt)
def replace_constants(expression):
    constants = {
        'pi': str(math.pi),
        'e': str(math.e),
    }

    for constant, value in constants.items():
        expression = expression.replace(constant, value)

    while 'sqrt(' in expression:
        start = expression.find('sqrt(')
        end = expression.find(')', start)
        sub_expression = expression[start+5:end]
        evaluated = math.sqrt(float(sub_expression))
        expression = expression[:start] + str(evaluated) + expression[end+1:]

    return expression

#   Function for evaluating the expression, if the user input has parenthesis and replacing constants into the converted constants
def evaluate_expression(expression):
    try:
        expression = replace_constants(expression)
        return eval(expression)
    except (ValueError, ZeroDivisionError, SyntaxError, NameError):
        return None
        
#   To handle errors
def evaluate_sub_expression(expression):
    try:
        return eval(expression)

    except (ValueError, ZeroDivisionError, SyntaxError, NameError):
        return None


def lagrange(x: list, y: list, p: float) -> float:

                # counts the length of the x
                n = len(x)

                # initializes value for the answer
                answer = 0.0000

                # Solving for the values of L(n)
                for i in range(n):
                    basis_i = y[i]
                    for j in range(n):
                        # prevents division by zero when i = j
                        if i != j:
                            # term = term * following operations
                            basis_i *= (p - x[j]) / (x[i] - x[j])          
                    answer += basis_i
                return answer

def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x_values = [a + i * h for i in range(n+1)]
    y_values = [f(x) for x in x_values]
    integral = (h / 2) * (y_values[0] + 2 * sum(y_values[1:n]) + y_values[n])
    return integral

def simpsons_rule(f, a, b, n):
    h = (b - a) / n
    x_values = [a + i * h for i in range(n+1)]
    y_values = [f(x) for x in x_values]

    # Check if the number of intervals is even
    if n % 2 != 0:
        raise ValueError("The number of intervals must be even for Simpson's rule.")

    # Apply Simpson's rule formula
    sum_odd = sum(y_values[1:n:2])
    sum_even = sum(y_values[2:n:2])
    integral = (h / 3) * (y_values[0] + 4 * sum_odd + 2 * sum_even + y_values[n])
    return integral

def eliminate_text_before_slash(string):
    index = string.find('/')
    if index != -1:
        return string[index+1:]
    return string

def split_fraction(text):
    # Remove spaces between operands and their preceding and succeeding operators
    text = re.sub(r'(?<=[+\-*/^])\s*(?=\d)|(?<=\d)\s*(?=[+\-*/^])\s*(?=\d)', '', text)

    # Split the expression by spaces
    tokens = re.split(r'\s+', text)
    tokens = [elem for i, elem in enumerate(tokens) if i % 2 != 1]
    tokens = [eliminate_text_before_slash(elem) for elem in tokens]
    return tokens


# Define an empty DataFrame
ddf = pd.DataFrame(columns=['Method', 'Function', 'Lower Limit', 'Upper Limit', 'Number of Intervals', 'Result'])

# Perform numerical integration and add the results to the DataFrame
def perform_integration(method, f, a, b, n):
    if method == 'Trapezoidal':
        result = trapezoidal_rule(f, a, b, n)
    elif method == 'Simpson':
        result = simpsons_rule(f, a, b, n)

    ddf.loc[len(ddf)] = [method, str(f), a, b, n, result]


#   Main function
def main():
    with st.sidebar:
        selected = option_menu(
            menu_title = "Main Menu", #required
            options= ["Machine Problem 1", "Machine Problem 2", "Machine Problem 3", "Machine Problem 4",]
        )

    if selected == "Machine Problem 1":
        tab1, tab2 = st.tabs(["Error Calculator", "MacLaurin"])
        with tab1:
            tab11, tab12, = st.tabs(["Calculator", "Guide"])
            with tab11:
                st.title("Error Calculator")
                st.write("Enter the expressions for the actual value and the approximated value.")

                actual_expression = st.text_input("Actual Value Expression")
                approx_expression = st.text_input("Approximated Value Expression")

                decimal_places = st.number_input("Number of Decimal Places", min_value=0, max_value=15, value=0)

                actual_value = evaluate_expression(actual_expression)
                approximated_value = evaluate_expression(approx_expression)

                if actual_value is not None:
                    st.write("Actual Value:", actual_value)

                if approximated_value is not None:
                    st.write("Approximated Value:", approximated_value)

                if st.button("Calculate") and actual_value is not None and approximated_value is not None:

                    st.markdown("## Rounded Output")
                    rounded_output = round(approximated_value, decimal_places)

                    absolute_error_rounded = calculate_absolute_error(actual_value, rounded_output)
                    relative_error_rounded = calculate_relative_error(actual_value, rounded_output)
                    
                    st.write("Rounded Approximated Value:", rounded_output)
                    st.write("Rounded Absolute Error:", absolute_error_rounded)
                    st.write("Rounded Relative Error:", f"{relative_error_rounded * 100}%")

                    st.markdown("## Chopped Output")
                    
                    chopped_output = math.trunc(approximated_value * (10 ** decimal_places)) / (10 ** decimal_places)

                    absolute_error_chopped = calculate_absolute_error(actual_value, chopped_output)
                    relative_error_chopped = calculate_relative_error(actual_value, chopped_output)

                    st.write("Chopped Approximated Value:", chopped_output)
                    st.write("Chopped Absolute Error:", absolute_error_chopped)
                    st.write("Chopped Relative Error:", f"{relative_error_chopped * 100}%")
            with tab12:
                st.header("Guide")
                st.subheader("Input Format")
                st.write("Use the format 'expression operator expression' for the actual value and the approximated value.")
                st.write("You can use parentheses to group sub-expressions.")
                st.write("Available commands: pi, e, sqrt")
                st.write("Example: (pi + 2) * sqrt(5) / 3")
                st.write("Max value for decimal places is 15.")

                st.subheader("Operators")
                st.write("Supported operators:")
                st.write("- Addition: `+`")
                st.write("- Subtraction: `-`")
                st.write("- Multiplication: `*`")
                st.write("- Division: `/`")
                st.write("- Power: `**`")
        with tab2:
            """
            # MacLaurin's Polynomial Calculator
            """
            tab21, tab22 = st.tabs(["Description", "Rules"])
            with tab21:
                """
                In this Calculator, you are asked to input
                * The Function
                * n (Degree of Taylor's Polynomial)
                * x (The number we use to evaluate)
                * No. of Decimal Places

                """
            with tab22:
                """
                * When writing pi, e, etc. type it with 'math.' behind it. (ex: math.pi, math.e, math.sqrt(num))
                * When writing exponents, rather than '^' use two asterisks. (ex: x**2)
                * Don't use foreign characters to not cause errors.
                * If you see a red error box, don't worry. Just finish filling out the information properly and the calculate button will appear.

                """
                    
            eks = 0

            f = st.text_input('Enter the function: ')
            n = st.number_input("Enter n for Taylor's Polynomial: ", min_value = 0, max_value = 100, step = 1)
            eks = (st.text_input("Enter x: "))
            dec = st.number_input("Enter no. of decimal places", min_value = 0, max_value = 100, step = 1)

            x = symbols('x')

            if eks:
                eks = eval(eks)

            #Creates an array to store the values of evaluated derivatives
            z = array('f', [])

            #Evaluates derivatives at 0
            fsym = 0

            if f:
                fsym = sympify(f)

            ini = N(fsym, subs={x: 0})
            z.append(ini)
            for i in range(n):
                if f and x:
                    dff = diff(f, x)
                sto = N(dff, subs={x: 0})
                z.append(sto)
                if f and x:
                    f = diff(f, x)

            #Plugs the results into McLaurin's Polynomial
            test = sum(x**i/factorial(i) * z[i] for i in range(n+1))

            #Makes decimal changes
            pnx = float(1)
            if test:
                pnx = N(test, subs={x: eks})

            #Chopping, Rounding, and Errors
            round = 0
            for i,c in enumerate(str(pnx)):
                if c == '.':
                    round = i + dec + 2
                    pnxr = float(str(pnx)[0:round])

            for i,c in enumerate(str(pnxr)):
                if i == (round-1):
                    if c == '5' or c == '6' or c == '7' or c == '8' or c == '9':
                        if pnx > 0:
                            pnxr = pnxr + (1*10**(-dec))
                        elif pnx < 0:
                            pnxr = pnxr - (1 * 10 ** (-dec))
                            
            pnxr = float(str(pnxr)[0:round-1])
            rerr = abs(pnx - float(pnxr))
            rrerr = abs((pnx - pnxr)/pnx) * 100

            chop = 0
            for i,c in enumerate(str(pnx)):
                if c == '.':
                    chop = i + dec + 1

            pnxc = float(str(pnx)[0:chop])
            cerr = abs(pnx - pnxc)
            crerr = abs((pnx-pnxc)/pnx) * 100

            if (st.button('Calculate     ')):
                st.write("Pn(x) = ", pnx)
                tab3, tab4 = st.tabs(["Rounding", "Chopping"])
                with tab3:
                    st.write("Rounded = ",pnxr)
                    st.write("Rounded Error = ",rerr)
                    st.write("Rounded Relative Error(in %) = ",rrerr)
                with tab4:
                    st.write("Chopped = ",pnxc)
                    st.write("Chopped Error = ",cerr)
                    st.write("Chopped Relative Error(in %) = ",crerr)


            print("")
            print("Pn(x) = ",pnx)
            print("")
            print("Rounded = ",pnxr)
            print("Rounded Error = ",rerr)
            print("Rounded Relative Error = ",rrerr,'%')
            print("")
            print("Chopped = ",pnxc)
            print("Chopped Error = ",cerr)
            print("Chopped Relative Error = ",crerr,'%')

            
            

    if selected == "Machine Problem 2":
        tab3, tab4= st.tabs(["Pre-defined", "User-defined"])
        
        with tab3:
            tab31, tab32 = st.tabs(["Bisection", "Secant"])
            with tab31:
                st.title("Bisection Method")
                st.text("Function: tanh(x)")
                a = st.number_input("Enter a", value = 0.0)
                b = st.number_input("Enter b", value = 0.0)

                choice = st.radio('Pick one', ['Iteration', 'Error'])
                if choice == 'Iteration':
                    count = 2
                    error = st.number_input("Enter Iterations", min_value=1)
                    st.write("Entered Values:")
                    st.write("A = ", a)
                    st.write("B = ", b)
                    st.write("Iterations = ", error)
                    c = (a+b)/2
                    d = math.tanh(c)
                    e = abs(a-b)
                    data = {
                        "a": [a], "b": [b], "c": [c], "f(c)": [d], "|a-b|": [e],
                    }
                    df = pd.DataFrame(data)

                    while count <= error:
                        if (math.tanh(a))*d > 0:
                            a = c
                        elif (math.tanh(b))*d > 0:
                            b = c

                        count += 1
                        c = (a+b)/2
                        d = math.tanh(c)
                        e = abs(a-b)

                        new_row = {'a': a, 'b': b, 'c': c, 'f(c)': d, '|a-b|':[e]}
                        df = pd.concat([df, pd.DataFrame(new_row)])

                    st.write("---")
                    st.subheader("Answers")
                    st.write("Cn= ", c)
                    st.write("f(Cn)= ", d)

                    # Display the dataframe in a table
                    st.write("---")
                    st.write("Table:")
                    df.index = np.arange(1, len(df) + 1)
                    st.table(df)
                    st.markdown("<span style='color:green'>Note: <br>the table can only show values up to 4 decimal places</span>", unsafe_allow_html=True)


                else:
                    error = st.number_input("Enter error", value = 0.01)
                    st.write("Entered Values:")
                    st.write("A = ", a)
                    st.write("B = ", b)
                    st.write("Error = ", error)
                    c = (a+b)/2
                    d = math.tanh(c)
                    e = abs(a-b)
                    data = {
                        "a": [a], "b": [b], "c": [c], "f(c)": [d], "|a-b|": [e],
                    }
                    df = pd.DataFrame(data)

                    while error <= e:
                        if (math.tanh(a))*d > 0:
                            a = c
                        elif (math.tanh(b))*d > 0:
                            b = c

                        c = (a+b)/2
                        d = math.tanh(c)
                        e = abs(a-b)

                        new_row = {'a': a, 'b': b, 'c': c, 'f(c)': d, '|a-b|':[e]}
                        df = pd.concat([df, pd.DataFrame(new_row)])

                    st.write("---")
                    st.subheader("Answers")
                    st.write("Cn= ", c)
                    st.write("f(Cn)= ", d)

                    # Display the dataframe in a table
                    st.write("---")
                    st.write("Table:")
                    df.index = np.arange(1, len(df) + 1)
                    
                    st.table(df)

                    st.markdown("<span style='color:green'>Note: <br>the table can only show values up to 4 decimal places</span>", unsafe_allow_html=True)


            with tab32:
                st.title("Secant Method")
                st.text("Function: tanh(x)")

                a = st.number_input("Enter X₀", value = 2, min_value=-2, max_value=2)
                b = st.number_input("Enter X₁", value=1, min_value=-2, max_value=2)
                st.markdown("<span style='color:red'>0 as an input for X₁ will not work on tanh(x) </span>", unsafe_allow_html=True)
                while abs(float(b)) == abs(float(a)):
                    st.warning("Inputs should not be equal to the absolute value of each other")
                    b = st.number_input("Enter X₁", value=1, min_value=-2, max_value=2)
                while (float(b)) == 0.0 or float(a) == 0:
                    st.warning("Please enter a non-zero value.")
                    b = st.number_input("Enter X₁", value=1, min_value=-2, max_value=2)


                choice = st.radio('Pick one ', ['Iteration', 'Error'])
                if choice == 'Iteration':
                        count = 2
                        error = st.number_input("Enter error ", min_value=1)
                        st.write("Entered Values:")
                        st.write("A = ", a)
                        st.write("B = ", b)
                        st.write("Iterations = ", error)
                        c = math.tanh(a)    
                        d = math.tanh(b)
                        e = (b - ((d)*(b-a)/(d-c)))
                        f = abs(e-b)
                        data = {
                            "Xi-1": [a], "Xi": [b], "f(Xi-1)": [c], "f(Xi)": [d], "Xi+1": [e], "|Xi+1 - Xi|": [f],
                        }
                        df = pd.DataFrame(data)

                        while count <= error:
                            tempa = b
                            tempb = e

                            a = tempa
                            b = tempb
                            count += 1
                            c = math.tanh(a)    
                            d = math.tanh(b) 
                            e = b - ((d)*(b-a)/(d-c))
                            f = abs(e-b)

                            new_row = {'Xi-1': a, 'Xi': b, 'f(Xi-1)': c, 'f(Xi)': d, 'Xi+1': e, '|Xi+1 - Xi|': [f]}
                            df = pd.concat([df, pd.DataFrame(new_row)])

                        st.write("---")
                        st.subheader("Answers")
                        st.write("Cn= ", e)
                        st.write("f(Cn)= ", math.tanh(e))

                        # Display the dataframe in a table
                        st.write("---")
                        st.write("Table:")
                        df.index = np.arange(1, len(df) + 1)
                        st.table(df)
                        st.markdown("<span style='color:green'>Note: <br>the table can only show values up to 4 decimal places</span>", unsafe_allow_html=True)


                else:
                    error = st.number_input("Enter error ",min_value=0.00001 , step=1e-5, format="%.5f")
                    st.write("Entered Values:")
                    st.write("A = ", a)
                    st.write("B = ", b)
                    st.write("Error = ", error)
                    c = math.tanh(a)    
                    d = math.tanh(b)
                    e = (b - ((d)*(b-a)/(d-c)))
                    f = abs(e-b)
                    data = {
                        "Xi-1": [a], "Xi": [b], "f(Xi-1)": [c], "f(Xi)": [d], "Xi+1": [e], "|Xi+1 - Xi|": [f],
                    }
                    df = pd.DataFrame(data)

                    while error <= f:
                        tempa = b
                        tempb = e

                        a = tempa
                        b = tempb
                        c = math.tanh(a)
                        d = math.tanh(b)
                        e = b - ((d)*(b-a)/(d-c))
                        f = abs(e-b)

                        new_row = {'Xi-1': a, 'Xi': b, 'f(Xi-1)': c, 'f(Xi)': d, 'Xi+1': e, '|Xi+1 - Xi|': [f]}
                        df = pd.concat([df, pd.DataFrame(new_row)])

                    st.write("---")
                    st.subheader("Answers")
                    st.write("Cn= ", e)
                    st.write("f(Cn)= ", math.tanh(e))

                    # Display the dataframe in a table
                    st.write("---")
                    st.write("Table:")
                    df.index = np.arange(1, len(df) + 1)
                    st.table(df)
                    st.markdown("<span style='color:green'>Note: <br>the table can only show values up to 4 decimal places</span>", unsafe_allow_html=True)

        with tab4:  
            x = symbols('x')
            e = symbols('e')
            pi = symbols('pi')
            print(math.pi)

            st.title("Bisection/Secant Method Calculator")
            fnb = st.text_input('Enter the function: ', value="x**2")
            a = st.number_input("Enter a", value=0.01)
            b = st.number_input("Enter b", value=0.01)

            if fnb:
                fbsym = sympify(fnb)

            fbsyme = fbsym.subs(e,math.e)
            fbsympi = fbsyme.subs(pi,math.pi)

            choice = st.radio('Choose one ', ['Iteration', 'Error'])
            if choice == 'Iteration':
                iter = st.number_input("Enter no. of iterations ", value=1)
            if choice == 'Error':
                error = st.number_input("Enter error. ", value = 0.000001)

            choice2= st.radio('Choose a method ', ['Bisection', 'Secant'])

            if (choice == 'Iteration') and choice2 == ('Bisection'):
                count = 2
                c = (a+b)/2
                d = N(fbsympi, subs={x: c})
                ee = abs(float(a)-(b))
                data = {
                    "a": [a],
                    "b": [b],
                    "c": [c],
                    "f(c)": [d],
                    "|a-b|": [ee]
                }
                df = pd.DataFrame(data)

                while count <= iter:
                    if N(fbsympi, subs={x: a})*d > 0:
                        a = c
                    elif N(fbsympi, subs={x: b})*d > 0:
                        b = c

                    count += 1
                    c = (a + b) / 2
                    d = N(fbsympi, subs={x: c})
                    ee = abs(a - b)

                    new_row = {'a': a, 'b': b, 'c': c, 'f(c)': d, '|a-b|':[ee]}
                    df = pd.concat([df, pd.DataFrame(new_row)])

                st.write("---")
                st.write("Cn= ", c)
                st.write("f(Cn)= ", d)

                # Display the dataframe in a table
                st.write("---")
                st.write("Table:")
                df.index = np.arange(1, len(df) + 1)
                st.write(df)

            elif (choice == 'Error') and choice2 == ('Bisection'):
                c = (a + b) / 2
                d = N(fbsympi, subs={x: c})
                ee = abs(a - b)
                data = {
                    "a": [a],
                    "b": [b],
                    "c": [c],
                    "f(c)": [d],
                    "|a-b|": [ee]
                }
                df = pd.DataFrame(data)

                while error <= ee:
                    if N(fbsympi, subs={x: a})*d > 0:
                        a = c
                    elif N(fbsympi, subs={x: b})*d > 0:
                        b = c

                    c = (a + b) / 2
                    d = N(fbsympi, subs={x: c})
                    ee = abs(a - b)

                    new_row = {'a': a, 'b': b, 'c': c, 'f(c)': d, '|a-b|':[ee]}
                    df = pd.concat([df, pd.DataFrame(new_row)])

                st.write("---")
                st.write("Error= ", error)
                st.write("Cn= ", c)
                st.write("f(Cn)= ", d)

                # Display the dataframe in a table
                st.write("---")
                st.write("Table:")
                df.index = np.arange(1, len(df) + 1)
                st.write(df)

            elif (choice == 'Iteration') and choice2 == ('Secant'):
                count = 2
                c = N(fbsympi, subs={x: a})
                d = N(fbsympi, subs={x: b})
                ee = (b - ((d) * (b - a) / (d - c)))
                f = abs(ee - b)
                data = {
                    "Xi-1": [a],
                    "Xi": [b],
                    "f(Xi-1)": [c],
                    "f(Xi)": [d],
                    "Xi+1": [ee],
                    "|Xi+1 - Xi|": [f],
                }
                df = pd.DataFrame(data)

                while count <= iter:
                    tempa = b
                    tempb = ee

                    a = tempa
                    b = tempb
                    count += 1
                    c = N(fbsympi, subs={x: a})
                    d = N(fbsympi, subs={x: b})
                    ee = (b - ((d) * (b - a) / (d - c)))
                    f = abs(ee - b)

                    new_row = {'Xi-1': a, 'Xi': b, 'f(Xi-1)': c, 'f(Xi)': d, 'Xi+1': ee, '|Xi+1 - Xi|': [f]}
                    df = pd.concat([df, pd.DataFrame(new_row)])

                st.write("---")
                st.write("Cn= ", c)
                st.write("f(Cn)= ", d)

                # Display the dataframe in a table
                st.write("---")
                st.write("Table:")
                df.index = np.arange(1, len(df) + 1)
                st.write(df)
            else:
                c = N(fbsympi, subs={x: a})
                d = N(fbsympi, subs={x: b})
                ee = (b - ((d)*(b-a)/(d-c)))
                f = abs(ee-b)
                data = {
                    "Xi-1": [a],
                    "Xi": [b],
                    "f(Xi-1)": [c],
                    "f(Xi)": [d],
                    "Xi+1": [ee],
                    "|Xi+1 - Xi|": [f],
                }
                df = pd.DataFrame(data)
                
                while error <= f:
                    tempa = b
                    tempb = ee

                    a = tempa
                    b = tempb
                    c = N(fbsympi, subs={x: a})
                    d = N(fbsympi, subs={x: b})
                    ee = (b - ((d) * (b - a) / (d - c)))
                    f = abs(ee - b)

                    new_row = {'Xi-1': a, 'Xi': b, 'f(Xi-1)': c, 'f(Xi)': d, 'Xi+1': ee, '|Xi+1 - Xi|': [f]}
                    df = pd.concat([df, pd.DataFrame(new_row)])

                st.write("---")
                st.write("Error= ", error)
                st.write("Cn= ", ee)
                st.write("f(Cn)= ", N(fbsympi, subs={x: ee}))

                # Display the dataframe in a table
                st.write("---")
                st.write("Table:")
                df.index = np.arange(1, len(df) + 1)
                st.write(df)

    if selected == "Machine Problem 3":
        tab5, tab6 = st.tabs(["Pre-defined", "User defined"])
        with tab5:
            st.title("Lagrange Interpolation Calculator")
            st.write("---")

            # header and link for stocks
            st.write("<h2 style='text-align: center;'> SMI Corp. Stocks</h2>", unsafe_allow_html= True)
            st.markdown("<p style='text-align: center; font-size: 16px;'><a href='https://www.marketwatch.com/investing/stock/sm/charts?countrycode=ph&mod=mw_quote_advanced'>SMI Corp. Stocks</a>", unsafe_allow_html=True)
            

            # input data points
            x_input = st.text_input("Enter x-coordinates (comma-separated):", value = "1,2,3,6,7,8,9,10,13,14,15,16,17,20,21", disabled=True) 
            y_input = st.text_input("Enter y-coordinates (comma-separated):", value = "889.000,890.500,896.000,886.000,880.000,889.000,871.000,875.000,875.000,860.000,874.000,874.000,897.000,883.500,900.000", disabled=True)

            # input value to interpolate at
            p_input = st.text_input("Enter x-value to interpolate at:", value = 0)

            # convert inputs to lists of floats
            x = []
            y = []
            p = None

            # tries to execute the following codes unless(except) it detects an error
            try:
                x = [float(i) for i in x_input.split(",")]
                y = [float(i) for i in y_input.split(",")]
                p = float(p_input)
            except ValueError:
                st.error("Please enter a valid value.")

            # calculate and display the interpolated value at P(x)
            if st.button("Calculate") and p is not None:
                answer = lagrange(x, y, p)
                st.write("---")
                # add the interpolated point to the input data
                x.append(p)
                y.append(answer)
                # display the interpolated value
                st.markdown(f"<br>The interpolated value at P({p}) is <span style='color:green'>{answer}</span>", unsafe_allow_html=True)
        
            st.write("<h5 style='text-align: center;'> Month of March</h5>", unsafe_allow_html= True)
            # create a pandas DataFrame from the input data
            data = pd.DataFrame({
                'x': x,
                'y': y
            })

            # Define y-axis range
            y_range = (data['y'].min() - 0.1, data['y'].max() + 0.1)

            # create a chart
            chart = alt.Chart(data).mark_line().encode(
            # To specify the range of values to display on the x and y axis
                x=alt.X('x', scale=alt.Scale(domain=(x[0], x[-1]))),
                y=alt.Y('y', scale=alt.Scale(domain=y_range))
            )

            # Add points
            points = chart.mark_circle(size=100, color='red').encode(
                x='x',
                y='y'
            )

            # Combine chart and points
            chart = (chart + points)

            # display the chart
            st.altair_chart(chart, use_container_width=True)

        with tab6:
            st.title("Lagrange Interpolation Calculator")
            st.write("---")

            # input data points (no fractions, only decimals)
            x_input = st.text_input("Enter x-coordinates (comma-separated): ", value = 0) 
            y_input = st.text_input("Enter y-coordinates (comma-separated): ", value = 0)

            # input value to interpolate at
            p_input = st.text_input("Enter x-value to interpolate at: ", value = 0)

            # convert inputs to lists of floats
            x = []
            y = []
            p = None

            # tries to execute the following codes unless(except) it detects an error
            try:
                x = [float(i) for i in x_input.split(",")]
                y = [float(i) for i in y_input.split(",")]
                p = float(p_input)
            except ValueError:
                st.error("Please enter a valid value.")

            # calculate and display the interpolated value at P(x)
            if st.button("Calculate ") and p is not None:
                answer = lagrange(x, y, p)
                st.write("---")
                # add the interpolated point to the input data
                x.append(p)
                y.append(answer)
                # display the interpolated value
                st.markdown(f"<br>The interpolated value at P({p}) is <span style='color:green'>{answer}</span>", unsafe_allow_html=True)
                
            # create a pandas DataFrame from the input data
            data = pd.DataFrame({'x': x, 'y': y})

            # create a chart
            chart = alt.Chart(data).mark_line().encode(
                x='x',
                y='y'
            )

            # Add points
            points = chart.mark_circle(size=100, color='red').encode(
                x='x',
                y='y'
            )

            # Combine chart and points
            chart = (chart + points)

            # display the chart
            st.altair_chart(chart, use_container_width=True)

    if selected == "Machine Problem 4":
        st.write("Supported operators:")
        st.write("Numerical Integration")

        tab7, tab8= st.tabs(["Pre-defined", "User-defined"])
        with tab7:
            tab71, tab72= st.tabs(["Trapezoidal", "Simpson"])
            with tab71:
                st.title("Numerical Integration with Simpson's Rule")

                # Function input
                st.subheader("Function to Integrate: tanh(x)")
                function_input = 'math.tanh(x)'
                
                # Limits of integration
                st.subheader("Limits of Integration")
                lower_limit = st.number_input(" Lower Limit", value=0.0)
                upper_limit = st.number_input(" Upper Limit", value=1.0)

                # Number of trapezoids
                st.subheader("Number of Trapezoids")
                num_trapezoids = st.number_input("Enter the number of trapezoids", value=100, step=1)

                # Calculate the approximate integral
                try:
                    lambda_function = eval("lambda x: " + function_input)
                    perform_integration('Trapezoidal', lambda_function, lower_limit, upper_limit, num_trapezoids)
                    st.subheader("Approximate Integral")
                    st.write(ddf['Result'].iloc[-1])
                except Exception as e:
                    st.subheader("Error")
                    st.write(str(e))

            with tab72:
                st.title("Numerical Integration with Simpson's Rule")

                # Function input
                st.subheader("Function to Integrate: math.tanh(x)")
                function_input = 'math.tanh(x)'
                
                # Limits of integration
                st.subheader("Limits of Integration")
                lower_limit = st.number_input(" Lower Limit ", value=0.0)
                upper_limit = st.number_input(" Upper Limit ", value=1.0)

                # Number of intervals
                st.subheader("Number of Intervals")
                num_intervals = st.number_input("Enter the number of intervals", value=100, step=1)

                # Calculate the approximate integral
                try:
                    lambda_function = eval("lambda x: " + function_input)
                    perform_integration('Simpson', lambda_function, lower_limit, upper_limit, num_intervals)
                    st.subheader("Approximate Integral")
                    st.write(ddf['Result'].iloc[-1])
                except Exception as e:
                    st.subheader("Error")
                    st.write(str(e))
                

        with tab8:
            tab81, tab82= st.tabs(["Trapezoidal", "Simpson"])
            with tab81:
                st.title("Numerical Integration with Trapezoidal Rule")

                # Function input
                st.subheader("Function to Integrate")
                function_input = st.text_input("Enter a function (e.g., x ** 2, math.tanh(x))", value="x ** 2")
                function_input = function_input.replace("e", "math.e")
                function_input = function_input.replace("pi", "math.pi")
                function_input = function_input.replace("^", "**")
        
                # Limits of integration
                st.subheader("Limits of Integration")
                lower_limit = st.number_input("Lower Limit", value=0.0)
                upper_limit = st.number_input("Upper Limit", value=1.0)

                a = lower_limit
                b = upper_limit
                # Checks to see if it is in quotient form then splits it if so 
                if('/' in function_input):
                    denoms = split_fraction(function_input)

                    x = symbols('x')
                    e = symbols('e')
                    pi = symbols('pi')
                    a = lower_limit
                    b = upper_limit
                    print(math.pi)

                    results = []
                    ctr = 0

                    for fnb in denoms:
                        if fnb:
                            fnb = fnb.replace("math.","")
                            fbsym = sympify(fnb)

                        fbsyme = fbsym.subs(e, math.e)
                        fbsympi = fbsyme.subs(pi, math.pi)

                        equation = Eq(fbsympi, 0)   
                        roots = solve((equation, x >= a, x <= b), x)
                        results.append(roots)

                    results = [x for x in results if x != False]

                    if(len(results) > 0):
                        st.write("There is an/are element/s inside [a,b] where the function is undefined. ")  
                        st.write("C: ", results)
                    else:
                        st.write("No roots.")

                        # Number of trapezoids
                        st.subheader("Number of Trapezoids")
                        num_trapezoids = st.number_input("Enter the number of trapezoids ", value=100, step=1)

                        # Calculate the approximate integral
                        try:
                            lambda_function = eval("lambda x: " + function_input)
                            perform_integration('Trapezoidal', lambda_function, lower_limit, upper_limit, num_trapezoids)
                            st.subheader("Approximate Integral")
                            st.write(ddf['Result'].iloc[-1])
                        except Exception as e:
                            st.subheader("Error")
                            st.write(str(e))
                else:
                    # Number of trapezoids
                    st.subheader("Number of Trapezoids")
                    num_trapezoids = st.number_input("Enter the number of trapezoids  ", value=100, step=1)

                    # Calculate the approximate integral
                    try:
                        lambda_function = eval("lambda x: " + function_input)
                        perform_integration('Trapezoidal', lambda_function, lower_limit, upper_limit, num_trapezoids)
                        st.subheader("Approximate Integral")
                        st.write(ddf['Result'].iloc[-1])
                    except Exception as e:
                        st.subheader("Error")
                        st.write(str(e))

            with tab82:
                st.subheader("Function to Integrate ")
                function_input = st.text_input("Enter a function (e.g., x ** 2, math.tanh(x)) ", value="x ** 2")
                function_input = function_input.replace("e", "math.e")
                function_input = function_input.replace("pi", "math.pi")

                # Limits of integration
                st.subheader("Limits of Integration ")
                lower_limit = st.number_input("Lower Limit ", value=0.0)
                upper_limit = st.number_input("Upper Limit ", value=1.0)

                a = lower_limit
                b = upper_limit
                # Checks to see if it is in quotient form then splits it if so 
                if('/' in function_input):
                    denoms = split_fraction(function_input)

                    x = symbols('x')
                    e = symbols('e')
                    pi = symbols('pi')
                    a = lower_limit
                    b = upper_limit
                    print(math.pi)

                    results = []
                    ctr = 0

                    for fnb in denoms:
                        if fnb:
                            fnb = fnb.replace("math.","")
                            fbsym = sympify(fnb)

                        fbsyme = fbsym.subs(e, math.e)
                        fbsympi = fbsyme.subs(pi, math.pi)

                        equation = Eq(fbsympi, 0)   
                        roots = solve((equation, x >= a, x <= b), x)
                        results.append(roots)

                    results = [x for x in results if x != False]

                    if(len(results) > 0):
                        st.write("There is an/are element/s inside [a,b] where the function is undefined. ")  
                        st.write("C: ", results)
                    else:
                        st.write("No roots.")
                    # Number of intervals
                    st.subheader("Number of Intervals")
                    num_intervals = st.number_input("Enter the number of intervals ", value=100, step=1)

                    # Calculate the approximate integral
                    try:
                        lambda_function = eval("lambda x: " + function_input)
                        perform_integration('Simpson', lambda_function, lower_limit, upper_limit, num_intervals)
                        st.subheader("Approximate Integral")
                        st.write(ddf['Result'].iloc[-1])
                    except Exception as e:
                        st.subheader("Error")
                        st.write(str(e))
                else:
                    # Number of intervals
                    st.subheader("Number of Intervals")
                    num_intervals = st.number_input("Enter the number of intervals ", value=100, step=1)

                    # Calculate the approximate integral
                    try:
                        lambda_function = eval("lambda x: " + function_input)
                        perform_integration('Simpson', lambda_function, lower_limit, upper_limit, num_intervals)
                        st.subheader("Approximate Integral")
                        st.write(ddf['Result'].iloc[-1])
                    except Exception as e:
                        st.subheader("Error")
                        st.write(str(e))


if __name__ == "__main__":
    main()

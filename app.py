import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import math
from fractions import Fraction
from streamlit_option_menu import option_menu
from sympy import *

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
#   Main function
def main():

    with st.sidebar:
        selected = option_menu(
            menu_title = "Main Menu", #required
            options= ["Machine Problem 1", "Machine Problem 2", "Machine Problem 3", "Machine Problem 4",]
        )

    if selected == "Machine Problem 1":
        
        tab1, tab2 = st.tabs(["Calculator", "Guide"])
        with tab1:
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
        with tab2:
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
        
    if selected == "Machine Problem 2":
        tab1, tab2= st.tabs(["Pre-defined", "User-defined"])
        
        with tab1:
            tab3, tab4 = st.tabs(["Bisection", "Secant"])
            with tab3:
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


            with tab4:
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

        with tab2:  
            tab3, tab4 = st.tabs(["Bisection", "Secant"])
            with tab3:
                st.write("---")

        
        
    if selected == "Machine Problem 3":

        tab1, tab2 = st.tabs(["Pre-defined", "User defined"])
        with tab1:
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

        with tab2:
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
        tab1, tab2= st.tabs(["Pre-defined", "User-defined"])
        
        with tab1:
            st.write("---")
        with tab2:
            st.write("---")
        
    

if __name__ == "__main__":
    main()

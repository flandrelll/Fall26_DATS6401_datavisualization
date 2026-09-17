#%%
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Data Visualization of Baseball Statistics")

df = pd.read_csv("HW1/BaseballHeightWeight.csv")

#%%
# brief introduction to the dataset
st.header("About this Dataset")
st.write("""This dataset contains the height and weight of baseball players. The data is visualized using a scatter plot to show the relationship between height and weight.""")
# %%
# preview
st.header("Data Preview")
st.write(df.head(10))

#%%
# chart
st.header("Scatter Plot of Height vs Weight")
fig, ax = plt.subplots()
ax.scatter(df['Height(inches)'], df['Weight(pounds)'], alpha=0.5)

ax.set_xlabel("Height (inches)")
ax.set_ylabel("Weight (pounds)")
ax.set_title("Scatter Plot of Height vs Weight")
st.pyplot(fig)

# chart explanation
st.header("Chart Explanation")
st.write("""The scatter plot above shows the relationship between the height and weight of baseball players. 
Each point represents a player, with their height on the x-axis and weight on the y-axis. 
The plot helps to visualize how height and weight are correlated among baseball players.""")

#%%
def fun_44(s):
    """
    Check whether the parentheses in the input string are valid
    The parentheses are valid if they are always in pairs and in correct order 
    You may assume the string only includes '(' and ')'
    
    Parameters
    ----------
    s : a string
    
    Returns
    ----------
    True or False : boolean
    """
    count = 0
    for char in s:
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
            if count < 0:
                return False
    if count == 0:
        return True
    else:
        return False

# Test
s_1 = "()"
s_2 = ")("
s_3 = "()()"
s_4 = "(())"

print(fun_44(s_1))
print(fun_44(s_2))
print(fun_44(s_3))
print(fun_44(s_4))
# %%
def fun_45(s):
    """
    Check whether the parentheses in the input string are valid
    The parentheses are valid if they are always in pairs and in correct order 
    You may assume the string includes not only '(' and ')', but also '[' and ']' and '{' and '}'   
    
    Parameters
    ----------
    s : a string
    
    Returns
    ----------
    True or False : boolean
    """
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}

    for char in s:
        if char in mapping.values():
            stack.append(char)
        elif char in mapping.keys():
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()

    if not stack:
        return True 
    else:
        return False

# Test
s_1 = "()"
s_2 = ")("
s_3 = "()()"
s_4 = "(())"

print(fun_45(s_1))
print(fun_45(s_2))
print(fun_45(s_3))
print(fun_45(s_4))
# %%
def fun_46(s):
    """
    Check whether the parentheses in the input string are valid
    The parentheses are valid if they are always in pairs and in correct order 
    You may assume the string includes not only '(' and ')', but also '[' and ']' and '{' and '}'   
    
    Parameters
    ----------
    s : a string
    
    Returns
    ----------
    True or False : boolean
    """
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}

    for char in s:
        if char in mapping.values():
            stack.append(char)
        elif char in mapping.keys():
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()

    if not stack:
        return True 
    else:
        return False
# %%
print("Hello from py")
# %%
def fun_48(tokens):
    """
    Evaluate an arithmetic expression in Reverse Polish notation
    You may assume all the nubmers in the expression are intergers
    
    Parameters
    ----------
    tokens : a list of strings
    
    Returns
    ----------
    the value of the arithmetic expression : a number
    0 if "divide by zero"
    """
    stack = []
    for token in tokens:
        if token in ['+', '-', '*', '/']:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                if b == 0:
                    return 0
                else:
                    stack.append(int(a / b))
        else:
            stack.append(int(token))
    return stack[0]

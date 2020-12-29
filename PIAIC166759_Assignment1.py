#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[2]:


import numpy as np


# 2. Create a null vector of size 10 

# In[9]:


arr10 = np.zeros((10))
# print("         THIS IS A NULL VECTOR OF SIZE '10'")
# print("         ==================================")
print(arr10)


# 3. Create a vector with values ranging from 10 to 49

# In[12]:


arr1 = np.arange(10,50)
# print("         THIS IS A VECTOR WITH VALUES RANGING FROM '10' TO '49'")
# print("         ======================================================")
print(arr1)


# 4. Find the shape of previous array in question 3

# In[255]:


# print("THE SHAPE OF THE VECTOR IS:")
# print("===========================")
print(arr1.shape)


# 5. Print the type of the previous array in question 3

# In[254]:


# print("TYPE OF THE VECTOR IS:")
# print("======================")
print(type(arr1))


# 6. Print the numpy version and the configuration
# 

# In[33]:


# print("NUMPY VERSION IS:")
# print("=================")
print(np.__version__)


# In[34]:


# print("NUMPY CONFIGURATION IS:")
# print("=======================")
print(np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[257]:


# print("DIMENSION OF ARRAY IS:")
# print("======================")
print(arr1.ndim)


# 8. Create a boolean array with all the True values

# In[258]:


# print("BOOLEAN ARRAY WITH ALL TRUE VALUES:")
# print("===================================")
boolean_array = np.ones((1,5),dtype = bool)
print(boolean_array)


# 9. Create a two dimensional array
# 
# 
# 

# In[3]:


# print("TWO DIMENSIONAL ARRAY:")
# print("======================")
arr2d = np.array(range(1,5)).reshape(2,2)
print(arr2d)


# 10. Create a three dimensional array
# 
# 

# In[253]:


# print("THREE DIMENSIONAL ARRAY:")
# print("========================")
arr3d = np.array(range(1,10)).reshape(3,3)
print(arr3d)


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[252]:


vector = np.arange(11)
# ORIGINAL ARRAY = [0,1,2,3,4,5,6,7,8,9,10]
# print("REVERSED ARRAY:")
# print("===============")
print(vector[::-1])


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[251]:


null_vector = np.zeros(10)
null_vector[4] = 1
# print("NULL VECTOR OF SIZE '10' WITH FIFTH VALUE '5':")
# print("==============================================")
print(null_vector)


# 13. Create a 3x3 identity matrix

# In[250]:


identity_matrix = np.ones((3,3))
# print("IDENTITY MATRIX OF SIZE '3x3':")
# print("==============================")
print(identity_matrix)


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[249]:


arr = np.array([1, 2, 3, 4, 5],dtype=float)
# print("ARRAY AFTER CONVERSION INTO FLOAT:")
# print("==================================")
print(arr)


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[248]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])
# print("'arr1' AND 'arr2' AFTER MULTIPLICATION:")
# print("=======================================")
print(arr1 * arr2)


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[247]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],

            [7., 2., 12.]])
arr3 = (arr1 > arr2)
# print("RESULTANT ARRAY:")
# print("================")
arr3 = arr3.astype('int64') 
print(arr3)


# 17. Extract all odd numbers from arr with values(0-9)

# In[246]:


arr = np.array(range(0,10))
# print("THE EXTRACTED ODD NUMBERS ARE:")
# print("==============================")
odd_num = arr[ arr % 2 != 0]
print(odd_num)


# 18. Replace all odd numbers to -1 from previous array

# In[245]:


arr = np.array(range(0,10))
# print("THE ARRAY AFTER REPLACING ODD NUMBERS WITH '-1':")
# print("================================================")
arr[ arr % 2 != 0] = -1
print(arr)


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[244]:


arr = np.arange(10)
# print("THE ARRAY AFTER REPLACING THE VALUES OF INDEXES 5,6,7,8 TO '12':")
# print("================================================================")
arr[5:9] = 12
print(arr)


# 20. Create a 2d array with 1 on the border and 0 inside

# In[243]:


pattern = np.ones((5,5))
pattern[1:-1,1:-1] = 0
# print("THIS IS AN 2D ARRAY WITH '1' ON THE BORDERS AND '0' INSIDE:")
# print("===========================================================")
print(pattern)


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[242]:


arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
# print("ARRAY AFTER REPLACING THE VALUE '5' TO '12':")
# print("============================================")
arr2d[1][1] = 12
print(arr2d)


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[241]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# print("THE FULL ARRAY AFTER CONVERTING THE ELEMENTS OF FIRST ARRAY TO '64':")
# print("====================================================================")
arr3d[0][0] = 64
print(arr3d)


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[240]:


arr2d = np.array(range(0,10)).reshape(2,5)
print(arr2d)
# print("THE FIRST 1D ARRAY AFTER SLICING:")
# print("=================================")
print(arr2d[0])


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[239]:


arr2d = np.array(range(0,10)).reshape(2,5)
print(arr2d)
# print("THE SECOND VALUE FROM SECOND 1D ARRAY:")
# print("======================================")
print(arr2d[1][1])


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[238]:


arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
# print("THE THIRD COLOUMN WITH ONLY FIRST TWO ROWS:")
# print("===========================================")
print(arr2d[1][1])


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[35]:


arr_rand = np.random.randn(100).reshape(10,10)
# print("THIS IS A '10x10' ARRAY WITH RANDOM VALUES:")
# print("===========================================")
print(arr_rand)


# In[138]:


# print("THE MAXIMUM VALUE IN THE PREVIOUS ARRAY IS:")
# print("===========================================")
print(arr_rand.max())


# In[139]:


# print("THE MINIMUM VALUE IN THE PREVIOUS ARRAY IS:")
# print("===========================================")
print(arr_rand.min())


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[237]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
# print("THESE ARE THE COMMON ELEMENTS BETWEEN 'a' and  'b':")
# print("===================================================")
print(a[a == b])


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[167]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
# print("THESE ARE THE POSITIONS WHERE THE ELEMENTS OF 'a' and 'b' MATCH:")
# print("================================================================")
print(np.where(a==b))


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[36]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
# print("THESE ARE ALL THE VALUES FROM THE ARRAY 'data' WHERE THE VALUES FROM ARRAY 'names' ARE NOT EQUAL TO 'Will':")
# print("===========================================================================================================")
print(data[names!='Will'])


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[28]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
mask = (names != 'Will') & (names == 'Joe') 
# print("THESE ARE ALL THE VALUES FROM THE ARRAY 'data' WHERE THE VALUES FROM ARRAY 'names' ARE NOT EQUAL TO 'Will' AND 'Joe':")
# print("=====================================================================================================================")
print(data[mask])


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[205]:


array = np.random.randint(low=1,high=15,size=(5,3))
# print("THIS IS AN 2D ARRAY OF SHAPE '5x3' WITH RANDOM NUMBERS BETWEEN '1' TO '15':")
# print("===========================================================================")
print(array)


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[207]:


array3 = np.random.randint(low=1,high=16,size=(2,2,4))
# print("THIS IS AN 2D ARRAY OF SHAPE '2,2,4' WITH RANDOM NUMBERS BETWEEN '1' TO '16':")
# print("===========================================================================")
print(array3)


# 33. Swap axes of the array you created in Question 32

# In[231]:


# print("SWAPPING THE AXES OF ABOVE ARRAY:")
# print("=================================")
print(np.swapaxes(array3,0,2))


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[236]:


arr10 = np.array(range(10))
# print("THIS IS THE ARRAY AFTER SQUARING:")
# print("=================================")
arr10squared = np.sqrt(arr10)
print(np.where(arr10squared<0.5,0,arr10squared))


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[30]:


arr1 = np.random.randn(12)
arr2 = np.random.randn(12)
maximum = np.maximum(arr1,arr2)
# print("THIS IS THE ARRAY WITH THE MAXIMUM VALUES BETWEEN EACH ELEMENT OF THE TWO ARRAYS:")
# print("=================================================================================")
print(maximum)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[235]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# print("THESE ARE THE UNIQUE NAMES AFTER SORTING:")
# print("=========================================")
unique = np.unique(names)
print(unique)


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[16]:


a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
result = np.setdiff1d(a, b)
# print("THIS ARRAY 'a' AFTER REMOVING ALL THE ELEMENTS PRESENT IN ARRAY 'b':")
# print("====================================================================")
print(result)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[232]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
print(sampleArray[1:-1])
# sampleArray[1]
sampleArray[:,1][:] = [10,10,10]
# print("THIS IS THE ARRAY AFTER CHANGING THE COLOUMN TWO:")
# print("=================================================")
print(sampleArray)


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[233]:


x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])
# print("DOT PRODUCT OF MATRIX 'x' AND 'y':")
# print("==================================")
print(np.dot(x,y))


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[234]:


matrix = np.random.randn(20)
commulative_sum = matrix.cumsum()
# print("COMMULATIVE SUM OF THE MATRIX:")
# print("==============================")
print(commulative_sum)


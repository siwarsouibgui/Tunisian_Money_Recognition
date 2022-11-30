{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f66d6228",
   "metadata": {},
   "outputs": [],
   "source": [
    "import  numpy as np \n",
    "import pandas as pd\n",
    "import math"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4184e6c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "csv_dataset_filename ='Tunisian_currency_Dataset.csv'\n",
    "model_filename = 'Tunisian_currency_recognition_model.h5'"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5bf1c817",
   "metadata": {},
   "source": [
    "# Data Preparation"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f8fc1ddc",
   "metadata": {},
   "source": [
    "the main steps of data preparation:\n",
    " - Normalization of data\n",
    " - convert categorical features\n",
    " - split data into test and train "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2cb58a5f",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = pd.read_csv()\n",
    "x = dataset.iloc[:,1:]\n",
    "y=dataset.iloc[:,0]\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "normalisation = MinMaxScaler()\n",
    "x= normalisation.fit_transform(x)\n",
    "\n",
    "y= pd.get_dummies(y).to_numpy()\n",
    "\n",
    "# data spliting\n",
    "from sklearn.model_selection import train_test_split\n",
    "x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "caab0d60",
   "metadata": {},
   "source": [
    "# Put model architecture"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9198e15c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import the libraries required for model creation\n",
    "\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense\n",
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3a77f5d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "model.add(Dense(11500,\"relu\",kernel_initializer=\"glorot_uniform\",input_dim = 22500))\n",
    "model.add(Dense(10000,\"relu\",kernel_initializer=\"glorot_uniform\"))\n",
    "model.add(Dense(8750,\"relu\",kernel_initializer=\"glorot_uniform\"))\n",
    "model.add(Dense(300,\"relu\",kernel_initializer=\"glorot_uniform\"))\n",
    "model.add(Dense(2,\"softmax\",kernel_initializer=\"glorot_uniform\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ba7613ad",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Compile the model\n",
    "model.compile(optimizer = \"adam\",loss = 'categorical_crossentropy',metrics =['accuracy'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b175d50f",
   "metadata": {},
   "source": [
    "Now let's train our model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7e6776fb",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.fit(x_train,y_train,batch_size = 10,epochs = 100)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "34ff3429",
   "metadata": {},
   "source": [
    "Our model is ready to test!!!"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "952a7ba6",
   "metadata": {},
   "source": [
    "# Test the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9340a4f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = model.predict(x_test)\n",
    "y_pred = (y_pred>0.5).astype(int)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cb15ddb6",
   "metadata": {},
   "source": [
    "Get accuracy of our model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b317c463",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import  accuracy_score\n",
    "accuracy_score(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1cb9a4d9",
   "metadata": {},
   "source": [
    "The last step is to save the model"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a05b88a1",
   "metadata": {},
   "source": [
    "# Save the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ec084592",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save(model_filename)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

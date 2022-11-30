{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "56464d72",
   "metadata": {},
   "source": [
    "# Import  libraries "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "75fdcf53",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import csv"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1697dbc4",
   "metadata": {},
   "source": [
    "# Preprocessing the dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c3f3f9ba",
   "metadata": {},
   "source": [
    "The first step is to convert paper money images to pixels and save them in a csv file.\n",
    "   - 1. Images should be resized to 80x80\n",
    "   - 2. Make a vector from out pixels\n",
    "   - 3. Insert the label index at the start of the vector\n",
    "   - 4. Save the data as a csv file"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "374ab6e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "path_to_dataset=\"Tunisian_currency_dataset/\"\n",
    "filename=\"Tunisian_currency_Dataset.csv\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4510694e",
   "metadata": {},
   "outputs": [],
   "source": [
    "labels = []\n",
    "first_img = True\n",
    "nb=1\n",
    "for label in os.listdir(path_to_dataset):\n",
    "    labels.append(label)\n",
    "for label in labels:\n",
    "    label_path = os.path.join(path,label)\n",
    "    for img_path in os.listdir(label_path):\n",
    "        new_path = os.path.join(label_path,img_path)\n",
    "        img = cv2.imread(new_path,cv2.IMREAD_GRAYSCALE)\n",
    "        img = cv2.resize(img,(80,80))\n",
    "        img_vector = np.empty((0,))\n",
    "        for i in range(len(img)):    \n",
    "            img_vector = np.r_[img_vector,img[i,:]]\n",
    "        img_vector = np.append(labels.index(label),img_vector)\n",
    "        img_vector =img_vector.reshape(-1,1).T\n",
    "        if first_img == True :\n",
    "            data = img_vector\n",
    "            first_img = False\n",
    "        data = np.r_[data,img_vector]\n",
    "        nb=nb+1\n",
    "     \n",
    "data = pd.DataFrame(data)\n",
    "data.to_csv(filename,index= False)"
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

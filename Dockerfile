# Start from official Python image
FROM python:3.12-slim

# Set a working directory in the container
WORKDIR /app

# Copy your notebook and README into the container
COPY Week_2_Mini-Assignment_Start_Your_First_Data_Analysis.ipynb README.md ./

# If you have a requirements.txt, copy it and install dependencies
# If you don't, you can install packages inline
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter Notebook
RUN pip install --no-cache-dir notebook pandas numpy matplotlib

# Expose the default Jupyter port
EXPOSE 8888

# Command to run Jupyter when the container starts
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
This is the link to the deployed app on streamlit. 
https://dat255project-epuctqwna74m3tqhhvscsg.streamlit.app/
In the report I also refference a local way of running the application.

Structure of the project:

comments_file.JSON - The file from where the streamlit app gathers the comments in order to predict them. Can be changed with new comments by following the structure used inside.

model.keras - The transformer model used for the predictions.

project_report.pdf - The project report.

requirements.txt - List of modules required for the streamlit app to deploy.

test.ipynb - The code used for the creation of the model.

transformer_encoder_model.keras - A backup for the model that I used to save the model. Not entirely necessary, but it's kept in case the actual model is lost.

visual_demonstration_streamlit.py - The code which is run to deploy the streamlit app.

vocab.pkl - The model's vocabulary that is used in the streamlit app.

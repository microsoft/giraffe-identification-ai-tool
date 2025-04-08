import os
import requests
import streamlit as st
from msal import ConfidentialClientApplication
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from dotenv import load_dotenv

load_dotenv()
authenticate_users, client_id, tenant_id, redirect_uri,key_vault_url, secret_name = map(os.getenv,
    ["authenticate_users", "client_id", "tenant_id", "redirect_uri", "key_vault_url", "secret_name"]
)

def authorize_users():
    """
    Returns the authentication configuration from configs.
    
    Returns:
        Function: The authenticate_users function from config_auth.
    """
    return authenticate_users == "True"


def get_secret(secret):
    """
    Retrieves a secret from Azure Key Vault.
    
    Args:
        secret (str): The name of the secret to retrieve.
        
    Returns:
        Secret: The requested secret from Azure Key Vault.
    """
    credential = DefaultAzureCredential()  
    secret_client = SecretClient(vault_url=key_vault_url, credential=credential)  
    return secret_client.get_secret(secret_name)  


def initialize_app():
    """
    Creates and initializes a Microsoft Authentication Library (MSAL) client application.
    
    Returns:
        ConfidentialClientApplication: Configured MSAL client for OAuth authentication.
    """
    client_secret = get_secret(secret_name).value
    authority_url = f"https://login.microsoftonline.com/{tenant_id}"
    return ConfidentialClientApplication(client_id, authority=authority_url, client_credential=client_secret)


def acquire_access_token(app, code, scopes, redirect_uri):
    """
    Exchanges an authorization code for an access token.
    
    Args:
        app (ConfidentialClientApplication): The MSAL client app.
        code (str): Authorization code from OAuth flow.
        scopes (list): List of permission scopes to request.
        redirect_uri (str): Redirect URI registered with the app.
        
    Returns:
        dict: Token response containing the access token and related information.
    """
    return app.acquire_token_by_authorization_code(code, scopes=scopes, redirect_uri=redirect_uri)


def fetch_user_data(access_token):
    """
    Retrieves user information from Microsoft Graph API.
    
    Args:
        access_token (str): OAuth access token for Microsoft Graph API.
        
    Returns:
        dict: User profile information from Microsoft Graph API.
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    graph_api_endpoint = "https://graph.microsoft.com/v1.0/me"
    response = requests.get(graph_api_endpoint, headers=headers)
    return response.json()

def authentication_process():
    """
    Manages the complete OAuth authentication flow using Microsoft identity platform.
    
    Handles authorization URL generation, redirect capture, token acquisition,
    and user profile fetching.
    
    Returns:
        dict: User data if authentication succeeds, None otherwise.
    """
    scopes = ["User.Read"]
    app = initialize_app()
    auth_url = app.get_authorization_request_url(scopes, redirect_uri=redirect_uri)
    if st.query_params.get("code"):
        st.session_state["auth_code"] = st.query_params.get("code")
        token_result = acquire_access_token(app, st.session_state.auth_code, scopes, redirect_uri)
        if "access_token" in token_result:
            user_data = fetch_user_data(token_result["access_token"])
            return user_data
        else:
            st.error("Failed to acquire token. Please check your input and try again.")
    else:
        st.markdown(f"Please go to [this URL]({auth_url}) and authorize the app.")



def login_ui():
    """
    Renders the login user interface and handles authentication state.
    
    Displays login prompt, initiates the authentication process, and updates
    the session state upon successful authentication.
    """
    if st.query_params.get("code"):
        st.title("Loading site...")
        user_data = authentication_process()
        if user_data:
            st.session_state["authenticated"] = True
            st.session_state["display_name"] = user_data.get("displayName")
            st.rerun()
    else:
        st.title("You need to sign in to use this app.")
        user_data = authentication_process()
        if user_data:
            st.session_state["authenticated"] = True
            st.session_state["display_name"] = user_data.get("displayName")
            st.rerun()
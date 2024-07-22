#!/bin/bash
#Experimental script in .sh to automate the process of automating the github interactions (pulling version)
# It is succesful but needs to be refined for more accurate control
# This script of course requires for git configurations to be properly setup and defined 
# Function to log messages
log_message() {
    echo "[Git Script]: $1"
}

# Function to check if a directory is a git repository
is_git_repo() {
    if git -C "$1" rev-parse 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to initialize a new git repository in case
initialize_git_repo() {
    local dir_name
    dir_name=$(basename "$1")
    local new_repo_name="${dir_name}RepoGit"

    log_message "Initializing a new Git repository: $new_repo_name"
    git init
    git remote add origin "$2"
    log_message "New repository initialized and remote set to: $2"
}

# Check for correct number of arguments
if [ "$#" -ne 2 ]; then
    log_message "Usage: $0 <repository-url> <local-directory>"
    exit 1
fi

REPO_URL=$1
LOCAL_DIR=$2
SSH_REPO_PATTERN="git@github.com"

# Change to the local directory
if cd "$LOCAL_DIR"; then
    log_message "Changed to directory: $LOCAL_DIR"
else
    log_message "Directory not found: $LOCAL_DIR"
    exit 1
fi

# Check if it's a git repository
if is_git_repo "$LOCAL_DIR"; then
    log_message "Updating existing repository..."
    git pull origin main || { log_message "Failed to pull from the repository. Check if the remote is set correctly."; exit 1; }
else
    log_message "Directory is not a Git repository. Setting it up..."
    if [[ "$REPO_URL" == *"$SSH_REPO_PATTERN"* ]]; then
        initialize_git_repo "$LOCAL_DIR" "$REPO_URL"
    else
        log_message "Cloning public repository using HTTPS..."
        git clone "$REPO_URL" .
    fi
fi



#Sample usage:
#                              Repository                    Local repository   |  
#                                                
#   SHH:
#
#  ./git_pull_script.sh "git@github.com:username/repo.git" "/path/to/local/dir"   Example: ./git_pull_script.sh "git@github.com:Ourple-Tech/Sabrine.git" "/Users/ali/Desktop/test" 
#
#  
#
#   HTTPS:
#
#   ./git_pull_script.sh "https://github.com/username/repository.git" "/path/to/local/dir"
#
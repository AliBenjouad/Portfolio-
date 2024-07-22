#!/bin/bash
#Experimental script in .sh to automate the process of automating the github interactions (pulling version)
# It is succesful but needs to be refined for more accurate control
# This script of course requires for git configurations to be properly setup and defined 
# Function to log messages
log_message() {
    echo "[Git Script]: $1"
}

# Function to log errors
log_error() {
    echo "[Git Script - ERROR]: $1" >&2
}

# Ensure we're not in a detached HEAD state
check_head_state() {
    if ! git symbolic-ref -q HEAD > /dev/null; then
        log_error "You are in a detached HEAD state. Please check out a branch before pushing."
        exit 1
    fi
}

# Function to handle git add operation with error checking
stage_files() {
    git add . 2>&1 | tee /tmp/git_add_output.txt
    if grep -q 'fatal:' /tmp/git_add_output.txt; then
        log_error "Failed to stage files. Please check the errors above."
        exit 1
    else
        log_message "Staged all files for commit."
    fi
    rm /tmp/git_add_output.txt
}

# Automatically handle initial commit and push for empty repositories
push_changes() {
    # Check if the repository is empty
    if [ -z "$(git rev-parse --quiet --verify HEAD)" ]; then
        log_message "The repository is empty. Preparing to make an initial commit."

        # Check for untracked files
        if [ -z "$(git ls-files --others --exclude-standard)" ]; then
            log_error "No files found to commit. Please add files to your repository."
            exit 1
        fi

        # Staging files
        stage_files

        # Making initial commit
        if ! git commit -m "Initial commit by script"; then
            log_error "Failed to make an initial commit."
            exit 1
        fi

        # Pushing to the remote repository
        if ! git push -u origin HEAD; then
            log_error "Failed to push initial commit to remote repository."
            exit 1
        else
            log_message "Pushed initial commit to remote repository successfully."
        fi
    else
        # Push changes if repository is not empty
        if ! git push origin HEAD; then
            log_error "Failed to push changes. Check your connection or if you have the right permissions."
            exit 1
        else
            log_message "Changes pushed successfully."
        fi
    fi
}

# Main script starts here

# Check for correct number of arguments
if [ "$#" -ne 1 ]; then
    log_error "Usage: $0 <path-to-project-directory>"
    exit 1
fi

PROJECT_DIR=$1

# Change to the project directory
if cd "$PROJECT_DIR"; then
    log_message "Changed to directory: $PROJECT_DIR"
else
    log_error "Directory not found: $PROJECT_DIR"
    exit 1
fi

# Ensure it's a Git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    log_error "This is not a git repository."
    exit 1
fi

check_head_state
push_changes













# sample usage:

#    path to the git push script         path of the target folder

# /Users/Ali/Desktop/git_push_script.sh /Users/Ali/Desktop/Sabrine-v1.0




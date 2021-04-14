## Common Submodule Workflow and Errors

Submodules in improv: CaImAn repository

## Useful Commands
1) To push to submodules when you push to master:

``
git push --recurse-submodules=on-demand
``

2) Get updates from upstream:

``
git submodule update --remote --merge
``

3) Add an automatic submodule update on merges

This allows submodules to be updated more frequently without notifications from team members.

``
echo "git submodule update --init --recursive" >> .git/hooks/post-merge
``

# Publishing to GitHub

## Suggested repository metadata

* **Name:** `bioastronautics-exosuit`
* **Description (one line):**
  Simulation and constrained optimisation of an active knee exoskeleton that compensates modelled EVA spacesuit joint resistance (Python, NumPy/SciPy/Matplotlib).
* **Topics:**
  `bioastronautics` `spacesuit` `eva` `exoskeleton` `biomechanics` `robotics` `kinematics` `optimization` `aerospace-engineering` `human-spaceflight` `scientific-computing` `python` `scipy` `matplotlib`

## Before publishing

Run these four checks first:

1. **Tests.** The suite should report 73 passed.

   ```bash
   python -m pytest
   ```

2. **README figures.** Regenerate the figures shown in the README.

   ```bash
   exosuit-sim --copy-figures-to docs/figures
   ```

3. **License.** Optionally, put your own name in the `LICENSE` copyright line.

4. **Secrets.** Check that nothing secret is staged. The project uses no credentials or API keys.

   ```bash
   git status
   ```

## Publish with the GitHub CLI

These commands assume `gh auth login` has already been run. Step 1 creates a local commit;
nothing is uploaded until step 2.

1. Create the initial commit. Skip this if the repository already has one.

   ```bash
   git init -b main
   ```

   ```bash
   git add .
   ```

   ```bash
   git commit -m "Initial commit: EVA suit exoskeleton simulation"
   ```

2. Create the public repository and push the commit to it.

   ```bash
   gh repo create bioastronautics-exosuit --public --source=. --remote=origin --push --description "Simulation and constrained optimisation of an active knee exoskeleton that compensates modelled EVA spacesuit joint resistance"
   ```

3. Add the topics.

   ```bash
   gh repo edit --add-topic bioastronautics,spacesuit,eva,exoskeleton,biomechanics,robotics,kinematics,optimization,aerospace-engineering,human-spaceflight,python
   ```

## Publish without the GitHub CLI

1. Create an **empty** repository named `bioastronautics-exosuit` on github.com, with no README, license, or `.gitignore`.
2. Add it as the remote. Replace `<your-username>` with your GitHub username.

   ```bash
   git remote add origin https://github.com/<your-username>/bioastronautics-exosuit.git
   ```

3. Push the `main` branch.

   ```bash
   git push -u origin main
   ```

`outputs/` is git-ignored, apart from `.gitkeep`. The README figures are tracked in
`docs/figures/` so they render on GitHub.

# 2048 — support de démonstration RL

Le projet contient maintenant un environnement Gymnasium de 2048 classique
(grille 4×4), avec PPO masquable. L’agent reçoit la grille sous forme de
valeurs `log2` normalisées et choisit l’une des quatre directions.

## Masquage d’actions

Une action est légale si elle modifie la grille. Avec `--mask-actions`,
`MaskablePPO` ne reçoit que ces directions. Sans masquage, les quatre actions
sont disponibles : une action inutile est pénalisée, mais ne fait pas
apparaître de nouvelle tuile. Cela rend la comparaison visible dans W&B via le
nombre et le taux d’actions invalides.

```bash
uv run rl-train --steps 1000000 --mask-actions \
    --wandb --wandb-project showcase
```

La sortie humaine affiche la dernière action et un petit triangle indiquant sa
direction. Elle démarre en pause : le bouton `Auto` active la lecture continue
et `Next action` demande exactement une décision supplémentaire. Le mode RGB
permet d’enregistrer une vidéo sans fenêtre.

Pour jouer manuellement :

```bash
uv run rl-manual
```

Les flèches et `WASD` déplacent les tuiles, `R` recommence une partie et
`Q`/`Échap` ferme la fenêtre. `--mask-actions` bloque les mouvements qui ne
peuvent pas modifier la grille.

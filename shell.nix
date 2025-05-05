{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-23.11.tar.gz") {} }:

pkgs.mkShell {
  buildInputs = [
    (pkgs.python3.withPackages (ps: with ps; [
      pandas
      numpy
      fastapi
      hydra-core
      pydantic
      scikit-learn
      joblib
      nltk
      mlflow
      rich
      boto3
      jupyter
      seaborn
      matplotlib
      spacy
      torch
      torchmetrics
      rapidfuzz
      pip
      setuptools
      rapidfuzz
      gensim
      transformers
      tqdm
      optuna
    ]))
  ];

  shellHook = ''
    jupyter notebook
  '';
}

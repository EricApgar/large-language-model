#!/usr/bin/env bash

CONFIG_TEMPLATE="config.yml.template"
CONFIG_FILE="config.yml"

if [ ! -f "$CONFIG_FILE" ]; then
    cp -n "$CONFIG_TEMPLATE" "$CONFIG_FILE"
    echo "The template has been copied. Please update $CONFIG_FILE with your details."
else
    echo "$CONFIG_FILE already exists."
fi
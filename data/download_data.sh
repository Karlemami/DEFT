#!/bin/bash

curl -O https://deft.lisn.upsaclay.fr/corpus/deft09_parlement_appr.xml.tar.gz
curl -O https://deft.lisn.upsaclay.fr/corpus/deft09_parlement_test.xml.tar.gz
curl -O https://deft.lisn.upsaclay.fr/corpus/deft09_parlement_ref.tar.gz


mkdir -p deft09_parlement_appr
mkdir -p deft09_parlement_test
mkdir -p deft09_parlement_ref

tar -xzf deft09_parlement_appr.xml.tar.gz -C deft09_parlement_appr
tar -xzf deft09_parlement_test.xml.tar.gz -C deft09_parlement_test
tar -xzf deft09_parlement_ref.tar.gz -C deft09_parlement_ref

rm deft09_parlement_appr.xml.tar.gz
rm deft09_parlement_test.xml.tar.gz
rm deft09_parlement_ref.tar.gz
import pytest
import requests

pytest_plugin = ["docker_compose"]


def test_homepage(homepage):
    assert requests.get(homepage).text.split("<h1>")[1].split(
        "</h1>")[0] == "Data Enginering Project"


def test_pred(homepage):

    response = requests.post(
        homepage+'/predict', data={'inputdata': 'This is great'})
    assert response.status_code == 200
    assert response.text.split("is ")[-1].split(".")[0] == 'positive'

    response = requests.post(
        homepage+'/predict', data={'inputdata': 'This is bad'})
    assert response.status_code == 200
    assert response.text.split("is ")[-1].split(".")[0] == 'negative'

    response = requests.post(
        homepage+'/predict', data={'inputdata': 'This is ok'})
    assert response.status_code == 200
    assert response.text.split("is ")[-1].split(".")[0] == 'neutral'

    response = requests.post(
        homepage+'/predict', data={'inputdata': "The song's lyrics are poor."})
    assert response.status_code == 200
    assert response.text.split("is ")[-1].split(".")[0] == 'negative'

    response = requests.post(
        homepage+'/predict', data={'inputdata': 'Cats are good pets, for they are clean.'})
    assert response.status_code == 200
    assert response.text.split("is ")[-1].split(".")[0] == 'positive'

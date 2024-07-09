import mujoco
import numpy as np


def check_contact(data, model, body1_name, body2_name):
    body1_id = data.body(body1_name).id
    body2_id = data.body(body2_name).id

    for i in range(data.ncon):
        try:
            contact = data.contact[i]
        except Exception as e:
            return False

        if (
            model.geom_bodyid[contact.geom1] == body1_id
            and model.geom_bodyid[contact.geom2] == body2_id
        ) or (
            model.geom_bodyid[contact.geom1] == body2_id
            and model.geom_bodyid[contact.geom2] == body1_id
        ):
            return True

    return False


def get_contact_force(data, model, body1_name, body2_name):
    body1_id = data.body(body1_name).id
    body2_id = data.body(body2_name).id

    contacts = []
    for i in range(data.ncon):
        try:
            contact = data.contact[i]
        except Exception as e:
            return 0

        if (
            model.geom_bodyid[contact.geom1] == body1_id
            and model.geom_bodyid[contact.geom2] == body2_id
        ) or (
            model.geom_bodyid[contact.geom1] == body2_id
            and model.geom_bodyid[contact.geom2] == body1_id
        ):
            contacts.append((i, contact))

    if len(contacts) == 0:
        return 0

    force = 0
    for i, con in contacts:
        c_array = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, i, c_array)
        force += np.linalg.norm(c_array)

    return force

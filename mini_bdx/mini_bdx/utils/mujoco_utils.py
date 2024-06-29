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

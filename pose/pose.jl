struct Point3
    x::Float64
    y::Float64
    z::Float64
end

Point3(tup::Tuple{U,U,U}) where {U<:Real} = Point3(tup[1], tup[2], tup[3])

Base.:+(a::Point3, b::Point3) = Point3(a.x + b.x, a.y + b.y, a.z + b.z)
Base.:-(a::Point3, b::Point3) = Point3(a.x - b.x, a.y - b.y, a.z - b.z)

tup(point::Point3) = (point.x, point.y, point.z)

struct BodyPose
    rotation::Point3
    elbow_r_loc::Point3
    elbow_l_loc::Point3
    elbow_r_rot::Point3
    elbow_l_rot::Point3
    hip_loc::Point3
    heel_r_loc::Point3
    heel_l_loc::Point3
end

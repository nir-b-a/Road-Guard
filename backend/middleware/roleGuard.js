const authorize = (...roles) => {
    return (req, res, next) => {
        if (!roles.includes(req.user.role)) {
            return res.status(403).json({ success: false, message: "Role '" + req.user.role + "' is not authorized for this route", data: null });
        }
        next();
    };
};
module.exports = { authorize };
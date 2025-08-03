import { Link } from "react-router-dom";
import ButtonSvg from "../assets/svg/ButtonSvg";

const Button = ({ className, href, to, onClick, children, px, white }) => {
  const classes = `button relative inline-flex items-center justify-center h-11 transition-all duration-300 hover:text-color-1 hover:scale-105 transform ${
    px || "px-7"
  } ${white ? "text-n-8" : "text-n-1"} ${className || ""}`;
  const spanClasses = "relative z-10";

  const renderButton = () => (
    <button className={classes} onClick={onClick}>
      <span className={spanClasses}>{children}</span>
      {ButtonSvg(white)}
    </button>
  );

  const renderLink = () => (
    <a href={href} className={classes}>
      <span className={spanClasses}>{children}</span>
      {ButtonSvg(white)}
    </a>
  );

  const renderRouterLink = () => (
    <Link to={to} className={classes}>
      <span className={spanClasses}>{children}</span>
      {ButtonSvg(white)}
    </Link>
  );

  if (to) return renderRouterLink();
  return href ? renderLink() : renderButton();
};

export default Button;

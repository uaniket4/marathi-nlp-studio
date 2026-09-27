import { Link } from 'react-router-dom'

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center">
      <p className="text-5xl font-semibold text-gray-300 dark:text-gray-700">404</p>
      <h1 className="mt-4 text-lg font-semibold text-gray-900 dark:text-gray-100">
        Page not found
      </h1>
      <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
        The page you’re looking for doesn’t exist.
      </p>
      <Link to="/" className="btn-primary mt-6">
        Back to Overview
      </Link>
    </div>
  )
}
